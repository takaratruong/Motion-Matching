#include "interaction_controller_adapter.h"
#include "interaction_target_rig_ik.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <utility>

namespace {

using interaction::FlatControllerPose;

struct FlatWorldPose {
    std::array<vec3, interaction::kFlatControllerBoneCount> positions{};
    std::array<quat, interaction::kFlatControllerBoneCount> rotations{};
};

quat normalized(quat value) {
    return value / quat_length(value);
}

FlatWorldPose flat_world(const FlatControllerPose& pose) {
    FlatWorldPose world{};
    for (size_t bone = 0; bone < interaction::kFlatControllerBoneCount;
         ++bone) {
        const int32_t parent = interaction::kFlatControllerParents[bone];
        if (parent < 0) {
            world.positions[bone] = pose.positions[bone];
            world.rotations[bone] = normalized(pose.rotations[bone]);
        } else {
            const size_t parent_bone = static_cast<size_t>(parent);
            world.positions[bone] = world.positions[parent_bone] +
                quat_mul_vec3(
                    world.rotations[parent_bone], pose.positions[bone]);
            world.rotations[bone] = normalized(quat_mul(
                world.rotations[parent_bone], pose.rotations[bone]));
        }
    }
    return world;
}

interaction::Pose interaction_reference() {
    interaction::Pose pose{};
    for (quat& rotation : pose.rotations) rotation = quat();
    return pose;
}

FlatControllerPose flat_reference() {
    FlatControllerPose pose{};
    for (size_t bone = 0; bone < pose.rotations.size(); ++bone) {
        pose.rotations[bone] = quat();
        const float value = static_cast<float>(bone + 1U);
        pose.velocities[bone] = vec3(
            0.01F * value, -0.02F * value, 0.03F * value);
        pose.angular_velocities[bone] = vec3(
            -0.04F * value, 0.05F * value, -0.06F * value);
    }
    pose.positions[15] = vec3(-0.15F, 0.0F, 0.0F);
    pose.positions[16] = vec3(-0.10F, 0.0F, 0.0F);
    pose.positions[17] = vec3(-0.30F, 0.0F, 0.0F);
    pose.positions[18] = vec3(-0.25F, 0.0F, 0.0F);
    pose.positions[19] = vec3(0.15F, 0.0F, 0.0F);
    pose.positions[20] = vec3(0.10F, 0.0F, 0.0F);
    pose.positions[21] = vec3(0.30F, 0.0F, 0.0F);
    pose.positions[22] = vec3(0.25F, 0.0F, 0.0F);
    return pose;
}

bool same_vec_bits(vec3 left, vec3 right) {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

bool same_quat_bits(quat left, quat right) {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

bool same_bits(const FlatControllerPose& left, const FlatControllerPose& right) {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

float rotation_error(quat left, quat right) {
    const float value = std::clamp(
        std::fabs(quat_dot(normalized(left), normalized(right))),
        0.0F,
        1.0F);
    return 2.0F * std::acos(value);
}

vec3 projected_direction(vec3 value, vec3 direction) {
    direction = normalize(direction);
    return normalize(value - direction * dot(value, direction));
}

void assert_finite_selected_channels(
    const FlatControllerPose& pose,
    interaction::Hand hand) {
    const size_t first = hand == interaction::Hand::Left ? 15U : 19U;
    for (size_t bone = first; bone < first + 4U; ++bone) {
        assert(std::isfinite(pose.rotations[bone].w));
        assert(std::isfinite(pose.rotations[bone].x));
        assert(std::isfinite(pose.rotations[bone].y));
        assert(std::isfinite(pose.rotations[bone].z));
        assert(std::isfinite(pose.angular_velocities[bone].x));
        assert(std::isfinite(pose.angular_velocities[bone].y));
        assert(std::isfinite(pose.angular_velocities[bone].z));
    }
}

void test_zero_weight_is_bit_exact() {
    const FlatControllerPose reference = flat_reference();
    FlatControllerPose pose = reference;
    interaction::TargetRigArmIK solver;
    solver.begin_epoch(
        interaction_reference(), reference, interaction::Hand::Right);
    const interaction::TargetRigArmIKResult result = solver.solve(
        pose,
        {vec3(0.3F, 0.4F, 0.2F), quat()},
        0.0F,
        1.0F / 60.0F);
    assert(!result.applied);
    assert(same_bits(pose, reference));
}

void test_reachable_target_keeps_clavicle_and_reaches_grasp() {
    FlatControllerPose pose = flat_reference();
    const FlatControllerPose before = pose;
    interaction::TargetRigArmIK solver;
    solver.begin_epoch(
        interaction_reference(), pose, interaction::Hand::Right);
    const quat target_rotation = quat_from_angle_axis(
        0.4F, normalize(vec3(0.2F, 0.9F, 0.3F)));
    const interaction::TargetRigArmIKResult result = solver.solve(
        pose,
        {vec3(0.45F, 0.25F, 0.0F), target_rotation},
        1.0F,
        1.0F / 60.0F);
    const FlatWorldPose world = flat_world(pose);
    assert(result.applied && result.reachable && !result.used_clavicle);
    assert(std::memcmp(
        &pose.rotations[19], &before.rotations[19], sizeof(quat)) == 0);
    assert(length(world.positions[22] - vec3(0.45F, 0.25F, 0.0F)) <=
           1.0e-3F);
    assert(rotation_error(world.rotations[22], target_rotation) <=
           0.5F * 3.14159265358979323846F / 180.0F);
}

void test_reachable_right_preserves_all_translations_and_inactive_channels() {
    FlatControllerPose pose = flat_reference();
    pose.foot_contacts = {1U, 0U};
    const FlatControllerPose before = pose;
    interaction::TargetRigArmIK solver;
    solver.begin_epoch(
        interaction_reference(), pose, interaction::Hand::Right);
    const interaction::TargetRigArmIKResult result = solver.solve(
        pose,
        {vec3(0.45F, 0.25F, 0.05F), quat()},
        1.0F,
        1.0F / 60.0F);
    assert(result.applied && result.reachable && !result.used_clavicle);

    for (size_t bone = 0; bone < pose.positions.size(); ++bone) {
        assert(same_vec_bits(pose.positions[bone], before.positions[bone]));
        assert(same_vec_bits(pose.velocities[bone], before.velocities[bone]));
    }
    for (size_t bone = 0; bone <= 18U; ++bone) {
        assert(same_quat_bits(pose.rotations[bone], before.rotations[bone]));
        assert(same_vec_bits(
            pose.angular_velocities[bone],
            before.angular_velocities[bone]));
    }
    assert(same_quat_bits(pose.rotations[19], before.rotations[19]));
    assert(same_vec_bits(
        pose.angular_velocities[19], before.angular_velocities[19]));
    assert(pose.foot_contacts == before.foot_contacts);
}

void test_nonidentity_epoch_calibration_is_rotation_only() {
    interaction::Pose source_reference = interaction_reference();
    const quat source_hand_rotation = quat_from_angle_axis(
        123.0F * 3.14159265358979323846F / 180.0F,
        normalize(vec3(0.3F, 0.8F, -0.2F)));
    source_reference.rotations[g1_skeleton::RightWrist] =
        source_hand_rotation;
    source_reference.positions[g1_skeleton::RightWrist] =
        vec3(40.0F, -30.0F, 20.0F);

    FlatControllerPose pose = flat_reference();
    pose.rotations[22] = quat_from_angle_axis(
        0.37F, normalize(vec3(-0.4F, 0.1F, 0.9F)));
    const quat flat_hand_reference = flat_world(pose).rotations[22];
    const quat expected_calibration = normalized(quat_inv_mul(
        source_hand_rotation, flat_hand_reference));
    interaction::TargetRigArmIK solver;
    solver.begin_epoch(
        source_reference, pose, interaction::Hand::Right);
    assert(rotation_error(
               solver.calibration_rotation(), expected_calibration) <
           1.0e-5F);

    const vec3 grasp_position(0.45F, 0.25F, 0.05F);
    const quat grasp_rotation = quat_from_angle_axis(
        0.61F, normalize(vec3(0.7F, -0.2F, 0.4F)));
    const interaction::TargetRigArmIKResult result = solver.solve(
        pose,
        {grasp_position, grasp_rotation},
        1.0F,
        1.0F / 60.0F);
    const FlatWorldPose solved = flat_world(pose);
    assert(result.applied && result.reachable);
    assert(length(solved.positions[22] - grasp_position) <= 1.0e-3F);
    assert(rotation_error(
               solved.rotations[22],
               normalized(quat_mul(
                   grasp_rotation, expected_calibration))) <=
           0.5F * 3.14159265358979323846F / 180.0F);
}

void test_intermediate_weight_reaches_blended_base_target() {
    FlatControllerPose pose = flat_reference();
    const FlatWorldPose base = flat_world(pose);
    interaction::TargetRigArmIK solver;
    solver.begin_epoch(
        interaction_reference(), pose, interaction::Hand::Right);
    const vec3 grasp_position(0.30F, 0.40F, 0.10F);
    const quat grasp_rotation = quat_from_angle_axis(
        0.8F, normalize(vec3(0.2F, 0.5F, 0.7F)));
    constexpr float weight = 0.35F;
    const vec3 expected_position = lerp(
        base.positions[22], grasp_position, weight);
    const quat expected_rotation = quat_nlerp_shortest(
        base.rotations[22], grasp_rotation, weight);
    const interaction::TargetRigArmIKResult result = solver.solve(
        pose,
        {grasp_position, grasp_rotation},
        weight,
        1.0F / 60.0F);
    const FlatWorldPose solved = flat_world(pose);
    assert(result.applied && result.reachable);
    assert(length(solved.positions[22] - expected_position) <= 1.0e-3F);
    assert(length(solved.positions[22] - grasp_position) > 0.10F);
    assert(rotation_error(solved.rotations[22], expected_rotation) <=
           0.5F * 3.14159265358979323846F / 180.0F);
}

void test_left_arm_is_mirrored_and_right_arm_remains_bit_exact() {
    FlatControllerPose pose = flat_reference();
    const FlatControllerPose before = pose;
    interaction::TargetRigArmIK solver;
    solver.begin_epoch(
        interaction_reference(), pose, interaction::Hand::Left);
    const vec3 left_grasp(-0.45F, 0.25F, 0.05F);
    const interaction::TargetRigArmIKResult result = solver.solve(
        pose,
        {left_grasp, quat()},
        1.0F,
        1.0F / 60.0F);
    const FlatWorldPose solved = flat_world(pose);
    assert(result.applied && result.reachable && !result.used_clavicle);
    assert(length(solved.positions[18] - left_grasp) <= 1.0e-3F);

    for (size_t bone = 0; bone < pose.positions.size(); ++bone) {
        assert(same_vec_bits(pose.positions[bone], before.positions[bone]));
        assert(same_vec_bits(pose.velocities[bone], before.velocities[bone]));
    }
    for (size_t bone = 0; bone <= 14U; ++bone) {
        assert(same_quat_bits(pose.rotations[bone], before.rotations[bone]));
        assert(same_vec_bits(
            pose.angular_velocities[bone],
            before.angular_velocities[bone]));
    }
    assert(same_quat_bits(pose.rotations[15], before.rotations[15]));
    assert(same_vec_bits(
        pose.angular_velocities[15], before.angular_velocities[15]));
    for (size_t bone = 19U; bone <= 22U; ++bone) {
        assert(same_quat_bits(pose.rotations[bone], before.rotations[bone]));
        assert(same_vec_bits(
            pose.angular_velocities[bone],
            before.angular_velocities[bone]));
    }
}

void test_clavicle_assist_and_unreachable_clamp_are_finite() {
    FlatControllerPose pose = flat_reference();
    interaction::TargetRigArmIK solver;
    solver.begin_epoch(
        interaction_reference(), pose, interaction::Hand::Right);
    const interaction::TargetRigArmIKResult assisted = solver.solve(
        pose,
        {vec3(0.75F, 0.24F, 0.0F), quat()},
        1.0F,
        1.0F / 60.0F);
    const FlatWorldPose assisted_world = flat_world(pose);
    assert(assisted.applied && assisted.reachable && assisted.used_clavicle);
    assert(length(assisted_world.positions[22] - vec3(0.75F, 0.24F, 0.0F)) <=
           1.0e-3F);

    pose = flat_reference();
    solver.begin_epoch(
        interaction_reference(), pose, interaction::Hand::Right);
    const interaction::TargetRigArmIKResult unreachable = solver.solve(
        pose,
        {vec3(1.0F, 0.0F, 0.0F), quat()},
        1.0F,
        1.0F / 60.0F);
    assert(unreachable.applied && !unreachable.reachable);
    assert(unreachable.reach_shortfall_m > 0.0F);
    for (const quat rotation : pose.rotations) {
        assert(std::isfinite(rotation.w));
        assert(std::isfinite(rotation.x));
        assert(std::isfinite(rotation.y));
        assert(std::isfinite(rotation.z));
    }
}

void test_previous_pole_rotates_with_spine_at_singularity() {
    interaction::TargetRigArmIK solver;
    FlatControllerPose seed_pose = flat_reference();
    solver.begin_epoch(
        interaction_reference(), seed_pose, interaction::Hand::Right);
    const vec3 seed_target(0.55F, 0.30F, 0.0F);
    const interaction::TargetRigArmIKResult seeded = solver.solve(
        seed_pose, {seed_target, quat()}, 1.0F, 1.0F / 60.0F);
    assert(seeded.applied && seeded.reachable);
    const FlatWorldPose seed_world = flat_world(seed_pose);
    const vec3 seed_direction =
        seed_world.positions[22] - seed_world.positions[20];
    const vec3 seed_pole = projected_direction(
        seed_world.positions[21] - seed_world.positions[20],
        seed_direction);

    FlatControllerPose turned_pose = flat_reference();
    const quat half_turn = quat_from_angle_axis(
        3.14159265358979323846F, vec3(0.0F, 0.0F, 1.0F));
    turned_pose.rotations[12] = half_turn;
    const FlatWorldPose turned_base = flat_world(turned_pose);
    const vec3 turned_direction = normalize(
        turned_base.positions[22] - turned_base.positions[20]);
    const vec3 turned_target =
        turned_base.positions[20] + 0.45F * turned_direction;
    const interaction::TargetRigArmIKResult turned = solver.solve(
        turned_pose, {turned_target, quat()}, 1.0F, 1.0F / 60.0F);
    assert(turned.applied && turned.reachable);
    const FlatWorldPose turned_world = flat_world(turned_pose);
    const vec3 turned_pole = projected_direction(
        turned_world.positions[21] - turned_world.positions[20],
        turned_world.positions[22] - turned_world.positions[20]);
    const vec3 expected_pole = projected_direction(
        quat_mul_vec3(half_turn, seed_pole),
        turned_world.positions[22] - turned_world.positions[20]);
    assert(dot(turned_pole, expected_pole) > 0.99F);
}

void test_straight_folded_and_antiparallel_targets_are_finite() {
    for (const interaction::Hand hand : {
             interaction::Hand::Left, interaction::Hand::Right}) {
        const size_t upper = hand == interaction::Hand::Left ? 16U : 20U;
        const size_t hand_bone = hand == interaction::Hand::Left ? 18U : 22U;
        const float side = hand == interaction::Hand::Left ? -1.0F : 1.0F;
        const std::array<float, 3> signed_distances = {
            0.55F, 0.05F, -0.40F};
        for (const float signed_distance : signed_distances) {
            FlatControllerPose pose = flat_reference();
            const FlatWorldPose base = flat_world(pose);
            const vec3 target = base.positions[upper] +
                vec3(side * signed_distance, 0.0F, 0.0F);
            interaction::TargetRigArmIK solver;
            solver.begin_epoch(interaction_reference(), pose, hand);
            const interaction::TargetRigArmIKResult result = solver.solve(
                pose, {target, quat()}, 1.0F, 1.0F / 60.0F);
            const FlatWorldPose solved = flat_world(pose);
            assert(result.applied && result.reachable);
            assert(std::isfinite(result.position_error_m));
            assert(std::isfinite(result.orientation_error_radians));
            assert(std::isfinite(result.reach_shortfall_m));
            assert(length(solved.positions[hand_bone] - target) <= 1.0e-3F);
            assert_finite_selected_channels(pose, hand);
        }
    }
}

void test_invalid_epoch_reset_and_invalid_requests_do_not_mutate() {
    interaction::TargetRigArmIK solver;
    assert(!solver.active());

    FlatControllerPose pose = flat_reference();
    const FlatControllerPose before = pose;
    interaction::Pose invalid_source = interaction_reference();
    invalid_source.rotations[g1_skeleton::RightWrist] =
        quat(0.0F, 0.0F, 0.0F, 0.0F);
    solver.begin_epoch(
        invalid_source, pose, interaction::Hand::Right);
    assert(!solver.active());
    assert(!solver.solve(
        pose, {vec3(0.4F, 0.2F, 0.0F), quat()},
        1.0F, 1.0F / 60.0F).applied);
    assert(same_bits(pose, before));

    FlatControllerPose invalid_flat_reference = before;
    invalid_flat_reference.rotations[22] =
        quat(0.0F, 0.0F, 0.0F, 0.0F);
    solver.begin_epoch(
        interaction_reference(),
        invalid_flat_reference,
        interaction::Hand::Right);
    assert(!solver.active());
    solver.begin_epoch(
        interaction_reference(),
        before,
        static_cast<interaction::Hand>(255U));
    assert(!solver.active());

    solver.begin_epoch(
        interaction_reference(), pose, interaction::Hand::Right);
    assert(solver.active());
    solver.reset();
    assert(!solver.active());
    assert(rotation_error(solver.calibration_rotation(), quat()) < 1.0e-6F);
    assert(!solver.solve(
        pose, {vec3(0.4F, 0.2F, 0.0F), quat()},
        1.0F, 1.0F / 60.0F).applied);
    assert(same_bits(pose, before));

    solver.begin_epoch(
        interaction_reference(), pose, interaction::Hand::Right);
    FlatControllerPose invalid_input_pose = before;
    invalid_input_pose.rotations[22] =
        quat(0.0F, 0.0F, 0.0F, 0.0F);
    const FlatControllerPose invalid_input_before = invalid_input_pose;
    assert(!solver.solve(
        invalid_input_pose,
        {vec3(0.4F, 0.2F, 0.0F), quat()},
        1.0F,
        1.0F / 60.0F).applied);
    assert(same_bits(invalid_input_pose, invalid_input_before));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<std::pair<interaction::Transform, float>, 4> requests = {{
        {{vec3(nan, 0.0F, 0.0F), quat()}, 1.0F / 60.0F},
        {{vec3(0.4F, 0.2F, 0.0F), quat(0.0F, 0.0F, 0.0F, 0.0F)},
         1.0F / 60.0F},
        {{vec3(0.4F, 0.2F, 0.0F), quat()}, 0.0F},
        {{vec3(0.4F, 0.2F, 0.0F), quat()}, nan},
    }};
    for (const auto& request : requests) {
        FlatControllerPose invalid_pose = before;
        const interaction::TargetRigArmIKResult result = solver.solve(
            invalid_pose, request.first, 1.0F, request.second);
        assert(!result.applied);
        assert(same_bits(invalid_pose, before));
    }
    for (const float invalid_weight : {-0.1F, 1.1F, nan}) {
        FlatControllerPose invalid_pose = before;
        const interaction::TargetRigArmIKResult result = solver.solve(
            invalid_pose,
            {vec3(0.4F, 0.2F, 0.0F), quat()},
            invalid_weight,
            1.0F / 60.0F);
        assert(!result.applied);
        assert(same_bits(invalid_pose, before));
    }
}

void test_segment_lengths_minimum_clavicle_swing_and_shortfall_semantics() {
    struct SolveCase {
        vec3 target;
        bool reachable;
        bool used_clavicle;
    };
    const std::array<SolveCase, 3> cases = {{
        {vec3(0.45F, 0.25F, 0.0F), true, false},
        {vec3(0.75F, 0.24F, 0.0F), true, true},
        {vec3(0.15F, 1.00F, 0.0F), false, true},
    }};
    for (const SolveCase& solve_case : cases) {
        FlatControllerPose pose = flat_reference();
        const FlatControllerPose before = pose;
        const FlatWorldPose base = flat_world(pose);
        interaction::TargetRigArmIK solver;
        solver.begin_epoch(
            interaction_reference(), pose, interaction::Hand::Right);
        const interaction::TargetRigArmIKResult result = solver.solve(
            pose, {solve_case.target, quat()}, 1.0F, 1.0F / 60.0F);
        const FlatWorldPose solved = flat_world(pose);
        assert(result.applied);
        assert(result.reachable == solve_case.reachable);
        assert(result.used_clavicle == solve_case.used_clavicle);
        for (size_t bone = 0; bone < pose.positions.size(); ++bone) {
            assert(same_vec_bits(pose.positions[bone], before.positions[bone]));
        }
        for (const std::pair<size_t, size_t>& segment : {
                 std::pair<size_t, size_t>{19U, 20U},
                 std::pair<size_t, size_t>{20U, 21U},
                 std::pair<size_t, size_t>{21U, 22U}}) {
            const float base_length = length(
                base.positions[segment.second] -
                base.positions[segment.first]);
            const float solved_length = length(
                solved.positions[segment.second] -
                solved.positions[segment.first]);
            assert(std::fabs(base_length - solved_length) <= 1.0e-6F);
        }
        if (solve_case.used_clavicle) {
            const vec3 before_offset =
                base.positions[20] - base.positions[19];
            const vec3 solved_offset =
                solved.positions[20] - solved.positions[19];
            const float minimum_swing = std::acos(std::clamp(
                dot(normalize(before_offset), normalize(solved_offset)),
                -1.0F,
                1.0F));
            assert(std::fabs(
                       rotation_error(
                           base.rotations[19], solved.rotations[19]) -
                       minimum_swing) <= 1.0e-4F);
        }
        if (!solve_case.reachable) {
            const float full_chain_length =
                length(base.positions[20] - base.positions[19]) +
                length(base.positions[21] - base.positions[20]) +
                length(base.positions[22] - base.positions[21]);
            const float expected_clamped_shortfall =
                length(solve_case.target - base.positions[19]) -
                (full_chain_length -
                 interaction::TargetRigArmIKConfig{}.reach_epsilon_m);
            assert(std::fabs(
                       result.reach_shortfall_m -
                       expected_clamped_shortfall) <= 1.0e-5F);
            assert(std::fabs(
                       result.reach_shortfall_m -
                       result.position_error_m) <= 1.0e-5F);
        }
    }
}

void test_selected_angular_velocities_follow_shortest_local_delta() {
    FlatControllerPose pose = flat_reference();
    const FlatControllerPose before = pose;
    interaction::TargetRigArmIK solver;
    solver.begin_epoch(
        interaction_reference(), pose, interaction::Hand::Right);
    constexpr float dt = 1.0F / 60.0F;
    const interaction::TargetRigArmIKResult result = solver.solve(
        pose,
        {vec3(0.35F, 0.28F, -0.18F),
         quat_from_angle_axis(
             2.8F, normalize(vec3(0.3F, -0.7F, 0.4F)))},
        1.0F,
        dt);
    assert(result.applied && result.reachable && !result.used_clavicle);
    for (size_t bone = 20U; bone <= 22U; ++bone) {
        assert(quat_dot(before.rotations[bone], pose.rotations[bone]) >= 0.0F);
        const float shortest_angle = rotation_error(
            before.rotations[bone], pose.rotations[bone]);
        assert(shortest_angle <= 3.14159265358979323846F + 1.0e-5F);
        assert(std::fabs(
                   length(pose.angular_velocities[bone]) * dt -
                   shortest_angle) <= 1.0e-4F);
    }
    assert_finite_selected_channels(pose, interaction::Hand::Right);
}

void test_reachable_small_angle_residual_is_not_physical_shortfall() {
    FlatControllerPose pose = flat_reference();
    interaction::TargetRigArmIK solver;
    solver.begin_epoch(
        interaction_reference(), pose, interaction::Hand::Right);

    const FlatWorldPose initial = flat_world(pose);
    const vec3 upper_root = initial.positions[20];
    const vec3 seed_target = upper_root + vec3(0.50F, 0.0F, 0.0F);
    const interaction::TargetRigArmIKResult seeded = solver.solve(
        pose, {seed_target, quat()}, 1.0F, 1.0F / 60.0F);
    assert(seeded.applied && seeded.reachable);

    constexpr float small_angle_radians = 0.002F;
    const vec3 perturbed_target = upper_root + quat_mul_vec3(
        quat_from_angle_axis(
            small_angle_radians, vec3(0.0F, 0.0F, 1.0F)),
        seed_target - upper_root);
    const interaction::TargetRigArmIKResult perturbed = solver.solve(
        pose, {perturbed_target, quat()}, 1.0F, 1.0F / 60.0F);

    assert(perturbed.applied);
    assert(perturbed.position_error_m > 2.0e-4F);
    assert(perturbed.position_error_m <= 1.0e-3F);
    assert(perturbed.reachable);
    assert(perturbed.reach_shortfall_m <= 1.0e-6F);
}

}  // namespace

int main() {
    test_zero_weight_is_bit_exact();
    test_reachable_target_keeps_clavicle_and_reaches_grasp();
    test_reachable_right_preserves_all_translations_and_inactive_channels();
    test_nonidentity_epoch_calibration_is_rotation_only();
    test_intermediate_weight_reaches_blended_base_target();
    test_left_arm_is_mirrored_and_right_arm_remains_bit_exact();
    test_clavicle_assist_and_unreachable_clamp_are_finite();
    test_previous_pole_rotates_with_spine_at_singularity();
    test_straight_folded_and_antiparallel_targets_are_finite();
    test_invalid_epoch_reset_and_invalid_requests_do_not_mutate();
    test_segment_lengths_minimum_clavicle_swing_and_shortfall_semantics();
    test_selected_angular_velocities_follow_shortest_local_delta();
    test_reachable_small_angle_residual_is_not_physical_shortfall();
    return 0;
}
