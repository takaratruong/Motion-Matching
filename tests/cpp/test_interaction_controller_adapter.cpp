#include "interaction_controller_adapter.h"
#include "locomotion_timing.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using interaction::ControllerInteractionEdges;
using interaction::ControllerInteractionFrameHandoff;
using interaction::ControllerInteractionHandConstraint;
using interaction::ControllerPlaceTarget;
using interaction::ControllerInteractionFrameState;
using interaction::ControllerInteractionSceneHandoff;
using interaction::ControllerInteractionSceneState;
using interaction::ControllerInteractionScheduler;
using interaction::Database;
using interaction::Features;
using interaction::FlatControllerPose;
using interaction::GraspAffordance;
using interaction::Hand;
using interaction::InteractionTarget;
using interaction::LocomotionSnapshot;
using interaction::ObjectState;
using interaction::Phase;
using interaction::PlaceStagingPreview;
using interaction::PickRequest;
using interaction::Pose;
using interaction::Reason;
using interaction::ResultCode;
using interaction::RuntimeInput;
using interaction::RuntimeOutput;
using interaction::RuntimeState;
using interaction::SurfaceHandle;
using interaction::TargetHandle;
using interaction::Transform;

struct FlatWorldPose {
    std::array<vec3, interaction::kFlatControllerBoneCount> positions{};
    std::array<vec3, interaction::kFlatControllerBoneCount> velocities{};
    std::array<quat, interaction::kFlatControllerBoneCount> rotations{};
    std::array<vec3, interaction::kFlatControllerBoneCount>
        angular_velocities{};
};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool float_bits_equal(float left, float right) {
    return std::memcmp(&left, &right, sizeof(float)) == 0;
}

bool vec_bits_equal(vec3 left, vec3 right) {
    return float_bits_equal(left.x, right.x) &&
           float_bits_equal(left.y, right.y) &&
           float_bits_equal(left.z, right.z);
}

bool quat_bits_equal(quat left, quat right) {
    return float_bits_equal(left.w, right.w) &&
           float_bits_equal(left.x, right.x) &&
           float_bits_equal(left.y, right.y) &&
           float_bits_equal(left.z, right.z);
}

bool pose_bits_equal(const Pose& left, const Pose& right) {
    for (size_t bone = 0; bone < left.positions.size(); ++bone) {
        if (!vec_bits_equal(left.positions[bone], right.positions[bone]) ||
            !vec_bits_equal(left.velocities[bone], right.velocities[bone]) ||
            !quat_bits_equal(left.rotations[bone], right.rotations[bone]) ||
            !vec_bits_equal(
                left.angular_velocities[bone],
                right.angular_velocities[bone])) {
            return false;
        }
    }
    for (size_t joint = 0; joint < left.hand_dof.size(); ++joint) {
        if (!float_bits_equal(left.hand_dof[joint], right.hand_dof[joint]) ||
            !float_bits_equal(
                left.hand_dof_velocities[joint],
                right.hand_dof_velocities[joint])) {
            return false;
        }
    }
    return left.foot_contacts == right.foot_contacts;
}

bool flat_pose_bits_equal(
    const FlatControllerPose& left,
    const FlatControllerPose& right) {
    for (size_t bone = 0; bone < left.positions.size(); ++bone) {
        if (!vec_bits_equal(left.positions[bone], right.positions[bone]) ||
            !vec_bits_equal(left.velocities[bone], right.velocities[bone]) ||
            !quat_bits_equal(left.rotations[bone], right.rotations[bone]) ||
            !vec_bits_equal(
                left.angular_velocities[bone],
                right.angular_velocities[bone])) {
            return false;
        }
    }
    return left.foot_contacts == right.foot_contacts;
}

bool frame_state_bits_equal(
    const ControllerInteractionFrameState& left,
    const ControllerInteractionFrameState& right) {
    const auto& left_result = left.hand_constraint_result;
    const auto& right_result = right.hand_constraint_result;
    return flat_pose_bits_equal(left.pose, right.pose) &&
        left_result.applied == right_result.applied &&
        left_result.reachable == right_result.reachable &&
        left_result.used_clavicle == right_result.used_clavicle &&
        float_bits_equal(
            left_result.position_error_m,
            right_result.position_error_m) &&
        float_bits_equal(
            left_result.orientation_error_radians,
            right_result.orientation_error_radians) &&
        float_bits_equal(
            left_result.reach_shortfall_m,
            right_result.reach_shortfall_m) &&
        left.hand_constraint_validated ==
            right.hand_constraint_validated &&
        quat_bits_equal(
            left.hand_constraint_calibration_rotation,
            right.hand_constraint_calibration_rotation) &&
        left.runtime_owns_pose == right.runtime_owns_pose &&
        left.overrides_locomotion_pose ==
            right.overrides_locomotion_pose &&
        left.synchronize_simulation_root ==
            right.synchronize_simulation_root &&
        vec_bits_equal(
            left.simulation_root_position,
            right.simulation_root_position) &&
        quat_bits_equal(
            left.simulation_root_rotation,
            right.simulation_root_rotation);
}

bool flat_bone_channels_bits_equal(
    const FlatControllerPose& left,
    const FlatControllerPose& right,
    size_t bone) {
    return vec_bits_equal(left.positions[bone], right.positions[bone]) &&
        vec_bits_equal(left.velocities[bone], right.velocities[bone]) &&
        quat_bits_equal(left.rotations[bone], right.rotations[bone]) &&
        vec_bits_equal(
            left.angular_velocities[bone], right.angular_velocities[bone]);
}

bool transform_bits_equal(const Transform& left, const Transform& right) {
    return vec_bits_equal(left.position, right.position) &&
           quat_bits_equal(left.rotation, right.rotation);
}

bool output_fields_equal(const RuntimeOutput& left, const RuntimeOutput& right) {
    const auto& left_diagnostics = left.diagnostics;
    const auto& right_diagnostics = right.diagnostics;
    if (left.owns_pose != right.owns_pose ||
        left.suppress_steering != right.suppress_steering ||
        !pose_bits_equal(left.pose, right.pose) ||
        !transform_bits_equal(left.object_world, right.object_world) ||
        left_diagnostics.state != right_diagnostics.state ||
        left_diagnostics.result != right_diagnostics.result ||
        left_diagnostics.reason != right_diagnostics.reason ||
        left_diagnostics.target != right_diagnostics.target ||
        left_diagnostics.object_state != right_diagnostics.object_state ||
        left_diagnostics.affordance_id != right_diagnostics.affordance_id ||
        left_diagnostics.clip != right_diagnostics.clip ||
        left_diagnostics.frame != right_diagnostics.frame ||
        left_diagnostics.phase != right_diagnostics.phase ||
        left_diagnostics.hand != right_diagnostics.hand ||
        !float_bits_equal(
            left_diagnostics.total_cost,
            right_diagnostics.total_cost) ||
        left_diagnostics.group_costs != right_diagnostics.group_costs ||
        !float_bits_equal(
            left_diagnostics.requested_root_correction_m,
            right_diagnostics.requested_root_correction_m) ||
        !float_bits_equal(
            left_diagnostics.applied_root_correction_m,
            right_diagnostics.applied_root_correction_m) ||
        !float_bits_equal(
            left_diagnostics.requested_yaw_correction_radians,
            right_diagnostics.requested_yaw_correction_radians) ||
        !float_bits_equal(
            left_diagnostics.applied_yaw_correction_radians,
            right_diagnostics.applied_yaw_correction_radians) ||
        !float_bits_equal(
            left_diagnostics.playback_speed,
            right_diagnostics.playback_speed) ||
        !float_bits_equal(
            left_diagnostics.hand_position_error_m,
            right_diagnostics.hand_position_error_m) ||
        !float_bits_equal(
            left_diagnostics.hand_orientation_error_radians,
            right_diagnostics.hand_orientation_error_radians) ||
        !float_bits_equal(
            left_diagnostics.hand_constraint_weight,
            right_diagnostics.hand_constraint_weight) ||
        left_diagnostics.attached != right_diagnostics.attached ||
        left_diagnostics.recorded_carry != right_diagnostics.recorded_carry ||
        left_diagnostics.inactive_arm_targets_locomotion !=
            right_diagnostics.inactive_arm_targets_locomotion ||
        left_diagnostics.inactive_arm_tracks_locomotion !=
            right_diagnostics.inactive_arm_tracks_locomotion ||
        left_diagnostics.pack_available != right_diagnostics.pack_available) {
        return false;
    }
    return true;
}

bool near(float left, float right, float tolerance = 1.0e-5F) {
    return std::fabs(left - right) <= tolerance;
}

void require_vec_near(
    vec3 actual,
    vec3 expected,
    const char* message,
    float tolerance = 1.0e-5F) {
    require(near(actual.x, expected.x, tolerance), message);
    require(near(actual.y, expected.y, tolerance), message);
    require(near(actual.z, expected.z, tolerance), message);
}

void require_same_rotation(
    quat actual,
    quat expected,
    const char* message,
    float tolerance = 1.0e-5F) {
    require(
        near(std::fabs(quat_dot(actual, expected)), 1.0F, tolerance),
        message);
}

float rotation_distance(quat left, quat right);

void assert_vec_near(vec3 actual, vec3 expected, float tolerance = 1.0e-5F) {
    assert(near(actual.x, expected.x, tolerance));
    assert(near(actual.y, expected.y, tolerance));
    assert(near(actual.z, expected.z, tolerance));
}

void assert_same_rotation(
    quat actual,
    quat expected,
    float tolerance = 1.0e-5F) {
    const float orientation_dot = std::fabs(quat_dot(actual, expected));
    assert(near(orientation_dot, 1.0F, tolerance));
}

Pose make_pose(float base, bool opposite_quaternion_sign = false) {
    Pose pose;
    for (size_t bone = 0; bone < pose.positions.size(); ++bone) {
        const float offset = static_cast<float>(bone) * 0.01F;
        pose.positions[bone] =
            vec3(base + offset, base + 1.0F + offset, base + 2.0F + offset);
        pose.velocities[bone] =
            vec3(base + 3.0F + offset, base + 4.0F, base + 5.0F - offset);
        const float half_angle = 0.25F + 0.001F * static_cast<float>(bone);
        quat rotation(cosf(half_angle), 0.0F, sinf(half_angle), 0.0F);
        pose.rotations[bone] = opposite_quaternion_sign ? -rotation : rotation;
        pose.angular_velocities[bone] =
            vec3(base + 6.0F, base + 7.0F + offset, base + 8.0F);
    }
    for (size_t joint = 0; joint < pose.hand_dof.size(); ++joint) {
        const float offset = static_cast<float>(joint) * 0.02F;
        pose.hand_dof[joint] = base + 9.0F + offset;
        pose.hand_dof_velocities[joint] = base + 10.0F - offset;
    }
    pose.foot_contacts = {
        static_cast<uint8_t>(base > 0.0F),
        static_cast<uint8_t>(base <= 0.0F)};
    return pose;
}

FlatControllerPose make_flat_pose(bool opposite_quaternion_sign = false) {
    FlatControllerPose pose;
    for (size_t bone = 0; bone < pose.positions.size(); ++bone) {
        const float value = static_cast<float>(bone + 1U);
        pose.positions[bone] = bone == 0U
            ? vec3(1.25F, 0.2F, -0.75F)
            : vec3(
                  0.015F * value,
                  0.04F + 0.006F * value,
                  -0.011F * value);
        pose.velocities[bone] = vec3(
            0.03F * value,
            -0.012F * value,
            0.017F * value);
        const vec3 axis = normalize(vec3(
            0.7F + 0.01F * value,
            0.4F + 0.02F * value,
            0.5F));
        quat rotation = quat_from_angle_axis(0.015F * value, axis);
        pose.rotations[bone] =
            opposite_quaternion_sign ? -rotation : rotation;
        pose.angular_velocities[bone] = vec3(
            -0.007F * value,
            0.009F * value,
            0.011F * value);
    }
    pose.foot_contacts = {1U, 0U};
    return pose;
}

Pose make_adversarial_true_g1_reference() {
    Pose pose = make_pose(0.25F);
    for (quat& rotation : pose.rotations) rotation = quat();
    pose.rotations[g1_skeleton::LeftHipPitch] = quat_from_angle_axis(
        3.141592654F, normalize(vec3(0.2F, 0.1F, 0.9F)));
    pose.rotations[g1_skeleton::RightHipPitch] = quat_from_angle_axis(
        -3.141592654F, normalize(vec3(0.1F, 0.3F, 0.9F)));
    pose.rotations[g1_skeleton::Spine] = quat_from_angle_axis(
        3.141592654F, normalize(vec3(0.7F, 0.2F, 0.4F)));
    pose.rotations[g1_skeleton::LeftShoulderPitch] = quat_from_angle_axis(
        3.141592654F, normalize(vec3(0.3F, 0.8F, 0.2F)));
    pose.rotations[g1_skeleton::RightShoulderPitch] = quat_from_angle_axis(
        -3.141592654F, normalize(vec3(0.4F, 0.7F, 0.2F)));
    return pose;
}

FlatWorldPose flat_world_pose(const FlatControllerPose& pose) {
    FlatWorldPose world;
    for (size_t bone = 0; bone < interaction::kFlatControllerBoneCount;
         ++bone) {
        const int32_t parent = interaction::kFlatControllerParents[bone];
        if (parent < 0) {
            world.positions[bone] = pose.positions[bone];
            world.velocities[bone] = pose.velocities[bone];
            world.rotations[bone] = pose.rotations[bone];
            world.angular_velocities[bone] = pose.angular_velocities[bone];
            continue;
        }
        const size_t parent_bone = static_cast<size_t>(parent);
        const vec3 offset = quat_mul_vec3(
            world.rotations[parent_bone], pose.positions[bone]);
        world.positions[bone] = world.positions[parent_bone] + offset;
        world.rotations[bone] = quat_normalize(quat_mul(
            world.rotations[parent_bone], pose.rotations[bone]));
        world.velocities[bone] =
            world.velocities[parent_bone] +
            cross(world.angular_velocities[parent_bone], offset) +
            quat_mul_vec3(
                world.rotations[parent_bone], pose.velocities[bone]);
        world.angular_velocities[bone] =
            world.angular_velocities[parent_bone] +
            quat_mul_vec3(
                world.rotations[parent_bone],
                pose.angular_velocities[bone]);
    }
    return world;
}

void assert_flat_pose_near(
    const FlatControllerPose& actual,
    const FlatControllerPose& expected,
    float tolerance = 2.0e-4F) {
    for (size_t bone = 0; bone < actual.positions.size(); ++bone) {
        assert_vec_near(actual.positions[bone], expected.positions[bone], tolerance);
        assert_vec_near(actual.velocities[bone], expected.velocities[bone], tolerance);
        assert_same_rotation(
            actual.rotations[bone], expected.rotations[bone], tolerance);
        assert_vec_near(
            actual.angular_velocities[bone],
            expected.angular_velocities[bone],
            tolerance);
    }
    assert(actual.foot_contacts == expected.foot_contacts);
}

void assert_pose_is_finite_and_unit(const Pose& pose) {
    for (size_t bone = 0; bone < pose.positions.size(); ++bone) {
        for (float value : {
                 pose.positions[bone].x,
                 pose.positions[bone].y,
                 pose.positions[bone].z,
                 pose.velocities[bone].x,
                 pose.velocities[bone].y,
                 pose.velocities[bone].z,
                 pose.rotations[bone].w,
                 pose.rotations[bone].x,
                 pose.rotations[bone].y,
                 pose.rotations[bone].z,
                 pose.angular_velocities[bone].x,
                 pose.angular_velocities[bone].y,
                 pose.angular_velocities[bone].z}) {
            assert(std::isfinite(value));
        }
        assert(near(quat_length(pose.rotations[bone]), 1.0F, 2.0e-5F));
    }
}

LocomotionSnapshot make_snapshot(float base) {
    LocomotionSnapshot snapshot;
    snapshot.pose = make_pose(base);
    for (size_t index = 0; index < snapshot.future_root_positions.size(); ++index) {
        const float value = base + static_cast<float>(index);
        snapshot.future_root_positions[index] = vec3(value, 0.0F, -value);
        snapshot.future_root_rotations[index] =
            quat_from_angle_axis(value * 0.1F, vec3(0.0F, 1.0F, 0.0F));
    }
    return snapshot;
}

RuntimeOutput make_complete_output(int serial, RuntimeState state) {
    RuntimeOutput output;
    output.owns_pose = (serial % 2) != 0;
    output.suppress_steering = (serial % 3) != 0;
    output.pose = make_pose(static_cast<float>(serial));
    output.object_world.position =
        vec3(static_cast<float>(serial), 2.0F, -3.0F);
    output.object_world.rotation =
        quat_from_angle_axis(0.2F * static_cast<float>(serial), vec3(0, 1, 0));
    output.diagnostics.state = state;
    output.diagnostics.result = ResultCode::Accepted;
    output.diagnostics.reason = Reason::None;
    output.diagnostics.target = {
        static_cast<uint64_t>(100 + serial),
        static_cast<uint32_t>(10 + serial)};
    output.diagnostics.object_state = ObjectState::Targeted;
    output.diagnostics.affordance_id = static_cast<uint32_t>(20 + serial);
    output.diagnostics.clip = serial;
    output.diagnostics.frame = 30 + serial;
    output.diagnostics.phase = Phase::Reach;
    output.diagnostics.hand = Hand::Left;
    output.diagnostics.total_cost = 1.0F + static_cast<float>(serial);
    output.diagnostics.group_costs = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    output.diagnostics.requested_root_correction_m = 0.11F;
    output.diagnostics.applied_root_correction_m = 0.09F;
    output.diagnostics.requested_yaw_correction_radians = 0.21F;
    output.diagnostics.applied_yaw_correction_radians = 0.19F;
    output.diagnostics.playback_speed = 1.05F;
    output.diagnostics.hand_position_error_m = 0.02F;
    output.diagnostics.hand_orientation_error_radians = 0.03F;
    output.diagnostics.hand_constraint_weight =
        0.01F * static_cast<float>(serial);
    output.diagnostics.attached = true;
    output.diagnostics.recorded_carry = true;
    output.diagnostics.inactive_arm_tracks_locomotion =
        (serial % 2) != 0;
    output.diagnostics.pack_available = true;
    return output;
}

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    assert(input.good());
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

#ifndef MM_REPO_SOURCE_ROOT
#error "MM_REPO_SOURCE_ROOT must be the compiled-in repository root"
#endif

std::string read_project_text(const std::string& relative_path) {
    return read_text(
        std::string(MM_REPO_SOURCE_ROOT) + "/" + relative_path);
}

size_t occurrence_count(
    const std::string& source,
    const std::string& needle) {
    require(!needle.empty(), "cannot count an empty source token");
    size_t count = 0U;
    size_t cursor = 0U;
    while ((cursor = source.find(needle, cursor)) != std::string::npos) {
        ++count;
        cursor += needle.size();
    }
    return count;
}

std::string without_ascii_whitespace(const std::string& source) {
    std::string compact;
    compact.reserve(source.size());
    for (unsigned char character : source) {
        if (!std::isspace(character)) {
            compact.push_back(static_cast<char>(character));
        }
    }
    return compact;
}

void test_exact_constants() {
    static_assert(
        locomotion_timing::kRateHz == 25,
        "flat locomotion must run at exactly 25 Hz");
    static_assert(
        locomotion_timing::kStepSeconds == 1.0F / 25.0F,
        "flat locomotion step must be exact binary32 1/25");
    static_assert(
        interaction::kControllerStepSeconds ==
            locomotion_timing::kStepSeconds,
        "controller and flat locomotion steps must be identical");
    static_assert(
        interaction::kInteractionRuntimeStepSeconds == 1.0F / 25.0F,
        "runtime step must be exact binary32 1/25");
    static_assert(
        locomotion_timing::kTrajectoryFrameOffsets[0] == 8 &&
            locomotion_timing::kTrajectoryFrameOffsets[1] == 17 &&
            locomotion_timing::kTrajectoryFrameOffsets[2] == 25,
        "trajectory feature offsets must be exact 25 Hz frame indices");
    static_assert(
        locomotion_timing::kTrajectorySampleTimesSeconds[0] == 0.32F &&
            locomotion_timing::kTrajectorySampleTimesSeconds[1] == 0.68F &&
            locomotion_timing::kTrajectorySampleTimesSeconds[2] == 1.0F,
        "trajectory prediction must use the feature sample times");
    static_assert(
        locomotion_timing::kTrajectoryStepSeconds[0] == 0.32F &&
            locomotion_timing::kTrajectoryStepSeconds[1] == 0.36F &&
            locomotion_timing::kTrajectoryStepSeconds[2] == 0.32F,
        "trajectory position prediction must use nonuniform step durations");
    static_assert(
        locomotion_timing::ticks_for_milliseconds(250U) == 7U &&
            locomotion_timing::ticks_for_milliseconds(500U) == 13U &&
            locomotion_timing::ticks_for_milliseconds(2000U) == 50U &&
            locomotion_timing::ticks_for_milliseconds(2500U) == 63U,
        "time-based controller windows must round up to 25 Hz ticks");
}

void test_flat_bridge_exact_parent_tree_anchor_map_and_unmapped_head() {
    static_assert(interaction::kFlatControllerBoneCount == 23U);
    static_assert(
        std::tuple_size<decltype(FlatControllerPose::positions)>::value == 23U);
    static_assert(
        std::tuple_size<decltype(FlatControllerPose::velocities)>::value == 23U);
    static_assert(
        std::tuple_size<decltype(FlatControllerPose::rotations)>::value == 23U);
    static_assert(
        std::tuple_size<
            decltype(FlatControllerPose::angular_velocities)>::value == 23U);

    constexpr std::array<int32_t, 23> expected_parents = {
        -1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10,
        11, 12, 13, 12, 15, 16, 17, 12, 19, 20, 21};
    assert(interaction::kFlatControllerParents == expected_parents);

    constexpr std::array<size_t, 21> expected_flat = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 15, 16, 17, 18, 19, 20, 21, 22};
    constexpr std::array<size_t, 21> expected_g1 = {
        0, 1, 2, 5, 6, 7, 8, 11, 12, 13, 14,
        15, 16, 17, 19, 20, 23, 24, 26, 27, 30};
    static_assert(interaction::kFlatControllerAnchors.size() == 21U);
    std::array<bool, interaction::kFlatControllerBoneCount> mapped_flat{};
    for (size_t index = 0; index < expected_flat.size(); ++index) {
        const interaction::FlatControllerAnchor anchor =
            interaction::kFlatControllerAnchors[index];
        assert(anchor.flat_bone == expected_flat[index]);
        assert(anchor.g1_bone == expected_g1[index]);
        assert(!mapped_flat[anchor.flat_bone]);
        mapped_flat[anchor.flat_bone] = true;
    }
    for (size_t bone = 0; bone < mapped_flat.size(); ++bone) {
        assert(mapped_flat[bone] == (bone != 13U && bone != 14U));
    }
    static_assert(interaction::kFlatControllerLeftToe == 5U);
    static_assert(interaction::kFlatControllerRightToe == 9U);
}

void test_flat_bridge_expansion_matches_every_world_anchor_and_retains_reference() {
    const FlatControllerPose flat = make_flat_pose();
    const Pose reference = make_pose(0.25F);
    const Pose expanded =
        interaction::expand_flat_controller_pose(flat, reference);
    assert_pose_is_finite_and_unit(expanded);

    const FlatWorldPose desired = flat_world_pose(flat);
    const interaction::WorldPose actual = interaction::world_pose(expanded);
    std::array<bool, g1_skeleton::BoneCount> mapped_g1{};
    for (const interaction::FlatControllerAnchor anchor :
         interaction::kFlatControllerAnchors) {
        mapped_g1[anchor.g1_bone] = true;
        assert_vec_near(
            actual.positions[anchor.g1_bone],
            desired.positions[anchor.flat_bone],
            3.0e-4F);
        assert_vec_near(
            actual.velocities[anchor.g1_bone],
            desired.velocities[anchor.flat_bone],
            3.0e-4F);
        assert_same_rotation(
            actual.rotations[anchor.g1_bone],
            desired.rotations[anchor.flat_bone],
            3.0e-4F);
        assert_vec_near(
            actual.angular_velocities[anchor.g1_bone],
            desired.angular_velocities[anchor.flat_bone],
            3.0e-4F);
    }

    for (size_t bone = 0; bone < mapped_g1.size(); ++bone) {
        if (mapped_g1[bone]) {
            continue;
        }
        assert(vec_bits_equal(expanded.positions[bone], reference.positions[bone]));
        assert(vec_bits_equal(expanded.velocities[bone], reference.velocities[bone]));
        assert(quat_bits_equal(expanded.rotations[bone], reference.rotations[bone]));
        assert(vec_bits_equal(
            expanded.angular_velocities[bone],
            reference.angular_velocities[bone]));
    }
    assert(expanded.hand_dof == reference.hand_dof);
    assert(expanded.hand_dof_velocities == reference.hand_dof_velocities);
    assert(expanded.foot_contacts == flat.foot_contacts);
}

void test_calibrated_flat_bridge_reference_is_exact_true_g1_pose() {
    const FlatControllerPose flat_reference = make_flat_pose();
    Pose interaction_reference = make_adversarial_true_g1_reference();
    interaction_reference.foot_contacts = flat_reference.foot_contacts;

    const Pose expanded = interaction::expand_flat_controller_pose(
        flat_reference, interaction_reference, flat_reference);

    require(
        pose_bits_equal(expanded, interaction_reference),
        "calibrated expansion changed its exact true-G1 reference pose");
}

void test_calibrated_flat_bridge_transfers_supported_world_deltas() {
    const FlatControllerPose flat_reference = make_flat_pose();
    FlatControllerPose flat_current = flat_reference;
    flat_current.positions[0] =
        flat_current.positions[0] + vec3(0.35F, -0.07F, 0.18F);
    flat_current.velocities[0] =
        flat_current.velocities[0] + vec3(-0.11F, 0.09F, 0.05F);
    for (size_t bone : {0U, 2U, 5U, 10U, 12U, 15U, 18U, 22U}) {
        const float value = static_cast<float>(bone + 1U);
        flat_current.rotations[bone] = quat_mul(
            quat_from_angle_axis(
                0.013F * value,
                normalize(vec3(0.4F, 0.7F, 0.2F))),
            flat_current.rotations[bone]);
        flat_current.angular_velocities[bone] =
            flat_current.angular_velocities[bone] +
            vec3(0.004F * value, -0.003F * value, 0.002F * value);
    }
    flat_current.foot_contacts = {0U, 1U};

    const Pose interaction_reference = make_adversarial_true_g1_reference();
    const Pose expanded = interaction::expand_flat_controller_pose(
        flat_current, interaction_reference, flat_reference);
    const FlatWorldPose flat_current_world = flat_world_pose(flat_current);
    const FlatWorldPose flat_reference_world = flat_world_pose(flat_reference);
    const interaction::WorldPose interaction_reference_world =
        interaction::world_pose(interaction_reference);
    const interaction::WorldPose expanded_world =
        interaction::world_pose(expanded);

    for (const interaction::FlatControllerAnchor anchor :
         interaction::kFlatControllerAnchors) {
        const quat flat_world_delta = quat_mul(
            flat_current_world.rotations[anchor.flat_bone],
            quat_inv(flat_reference_world.rotations[anchor.flat_bone]));
        const quat expected_rotation = quat_mul(
            flat_world_delta,
            interaction_reference_world.rotations[anchor.g1_bone]);
        require_same_rotation(
            expanded_world.rotations[anchor.g1_bone],
            expected_rotation,
            "calibrated expansion lost a mapped world-rotation delta",
            3.0e-4F);
        require_vec_near(
            expanded_world.angular_velocities[anchor.g1_bone],
            interaction_reference_world.angular_velocities[anchor.g1_bone] +
                flat_current_world.angular_velocities[anchor.flat_bone] -
                flat_reference_world.angular_velocities[anchor.flat_bone],
            "calibrated expansion lost a mapped world-angular delta",
            4.0e-4F);
    }

    require_vec_near(
        expanded.positions[0],
        interaction_reference.positions[0] + flat_current.positions[0] -
            flat_reference.positions[0],
        "calibrated expansion lost the root-position delta");
    require_vec_near(
        expanded.velocities[0],
        interaction_reference.velocities[0] + flat_current.velocities[0] -
            flat_reference.velocities[0],
        "calibrated expansion lost the root-velocity delta");

    std::array<bool, g1_skeleton::BoneCount> mapped{};
    for (const interaction::FlatControllerAnchor anchor :
         interaction::kFlatControllerAnchors) {
        mapped[anchor.g1_bone] = true;
    }
    for (size_t bone = 1; bone < g1_skeleton::BoneCount; ++bone) {
        require(
            vec_bits_equal(
                expanded.positions[bone],
                interaction_reference.positions[bone]) &&
                vec_bits_equal(
                    expanded.velocities[bone],
                    interaction_reference.velocities[bone]),
            "calibrated expansion warped true-G1 non-root morphology");
        if (!mapped[bone]) {
            require(
                quat_bits_equal(
                    expanded.rotations[bone],
                    interaction_reference.rotations[bone]) &&
                    vec_bits_equal(
                        expanded.angular_velocities[bone],
                        interaction_reference.angular_velocities[bone]),
                "calibrated expansion changed an unmapped true-G1 channel");
        }
    }
    require(
        expanded.hand_dof == interaction_reference.hand_dof &&
            expanded.hand_dof_velocities ==
                interaction_reference.hand_dof_velocities &&
            expanded.foot_contacts == flat_current.foot_contacts,
        "calibrated expansion changed hand channels or lost contacts");
}

void test_calibrated_flat_bridge_round_trips_representable_flat_pose() {
    const FlatControllerPose flat_reference = make_flat_pose();
    FlatControllerPose flat_current = flat_reference;
    flat_current.positions[0] =
        flat_current.positions[0] + vec3(-0.21F, 0.04F, 0.13F);
    flat_current.velocities[0] =
        flat_current.velocities[0] + vec3(0.07F, -0.03F, 0.09F);
    for (const interaction::FlatControllerAnchor anchor :
         interaction::kFlatControllerAnchors) {
        const size_t bone = anchor.flat_bone;
        const float value = static_cast<float>(bone + 1U);
        flat_current.rotations[bone] = quat_mul(
            quat_from_angle_axis(
                0.004F * value,
                normalize(vec3(0.3F, 0.8F, 0.5F))),
            flat_current.rotations[bone]);
        flat_current.angular_velocities[bone] =
            flat_current.angular_velocities[bone] +
            vec3(0.001F * value, -0.002F * value, 0.003F * value);
    }
    flat_current.foot_contacts = {0U, 1U};
    const Pose interaction_reference = make_adversarial_true_g1_reference();

    const Pose expanded = interaction::expand_flat_controller_pose(
        flat_current, interaction_reference, flat_reference);
    const FlatControllerPose collapsed = interaction::collapse_interaction_pose(
        expanded, interaction_reference, flat_reference);

    assert_flat_pose_near(collapsed, flat_current, 6.0e-4F);
    for (const size_t unmapped_flat_bone : {13U, 14U}) {
        require(
            quat_bits_equal(
                collapsed.rotations[unmapped_flat_bone],
                flat_reference.rotations[unmapped_flat_bone]) &&
                vec_bits_equal(
                    collapsed.angular_velocities[unmapped_flat_bone],
                    flat_reference.angular_velocities[unmapped_flat_bone]),
            "representable round trip changed an unmapped flat channel");
    }
}

void test_calibrated_flat_bridge_projects_nonrepresentable_flat_morphology() {
    const FlatControllerPose flat_reference = make_flat_pose();
    FlatControllerPose flat_current = flat_reference;
    flat_current.positions[5] =
        flat_current.positions[5] + vec3(3.0F, -2.0F, 4.0F);
    flat_current.velocities[18] =
        flat_current.velocities[18] + vec3(-5.0F, 6.0F, 7.0F);
    const Pose interaction_reference = make_adversarial_true_g1_reference();

    const Pose expanded = interaction::expand_flat_controller_pose(
        flat_current, interaction_reference, flat_reference);
    for (size_t bone = 1; bone < g1_skeleton::BoneCount; ++bone) {
        require(
            vec_bits_equal(
                expanded.positions[bone],
                interaction_reference.positions[bone]) &&
                vec_bits_equal(
                    expanded.velocities[bone],
                    interaction_reference.velocities[bone]),
            "nonrepresentable flat translations warped true-G1 morphology");
    }
    const FlatControllerPose projected = interaction::collapse_interaction_pose(
        expanded, interaction_reference, flat_reference);
    require(
        vec_bits_equal(
            projected.positions[5], flat_reference.positions[5]) &&
            vec_bits_equal(
                projected.velocities[18], flat_reference.velocities[18]),
        "calibrated bridge did not explicitly project unsupported flat channels");
}

void test_calibrated_flat_bridge_is_antipodal_invariant() {
    FlatControllerPose flat_reference = make_flat_pose();
    FlatControllerPose flat_current = flat_reference;
    flat_current.rotations[12] = quat_mul(
        quat_from_angle_axis(
            0.17F, normalize(vec3(0.2F, 0.9F, 0.3F))),
        flat_current.rotations[12]);
    Pose interaction_reference = make_adversarial_true_g1_reference();

    FlatControllerPose antipodal_current = flat_current;
    FlatControllerPose antipodal_flat_reference = flat_reference;
    Pose antipodal_interaction_reference = interaction_reference;
    for (quat& rotation : antipodal_current.rotations) rotation = -rotation;
    for (quat& rotation : antipodal_flat_reference.rotations) {
        rotation = -rotation;
    }
    for (quat& rotation : antipodal_interaction_reference.rotations) {
        rotation = -rotation;
    }

    const Pose positive = interaction::expand_flat_controller_pose(
        flat_current, interaction_reference, flat_reference);
    const Pose antipodal = interaction::expand_flat_controller_pose(
        antipodal_current,
        antipodal_interaction_reference,
        antipodal_flat_reference);
    const interaction::WorldPose positive_world = interaction::world_pose(
        positive);
    const interaction::WorldPose antipodal_world = interaction::world_pose(
        antipodal);
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        require_vec_near(
            antipodal_world.positions[bone],
            positive_world.positions[bone],
            "antipodal calibrated expansion changed a world position",
            4.0e-4F);
        require_same_rotation(
            antipodal_world.rotations[bone],
            positive_world.rotations[bone],
            "antipodal calibrated expansion changed a world rotation",
            4.0e-4F);
        require_vec_near(
            antipodal_world.angular_velocities[bone],
            positive_world.angular_velocities[bone],
            "antipodal calibrated expansion changed angular velocity",
            4.0e-4F);
    }
}

void test_calibrated_flat_bridge_rejects_invalid_input_without_mutation() {
    const auto expect_format_error = [](const auto& operation) {
        bool threw = false;
        try {
            operation();
        } catch (const interaction::FormatError&) {
            threw = true;
        }
        require(threw, "calibrated expansion accepted invalid input");
    };
    FlatControllerPose flat_current = make_flat_pose();
    Pose interaction_reference = make_adversarial_true_g1_reference();
    FlatControllerPose flat_reference = make_flat_pose();

    flat_current.positions[8].z = std::numeric_limits<float>::quiet_NaN();
    const FlatControllerPose flat_current_before = flat_current;
    expect_format_error([&] {
        (void)interaction::expand_flat_controller_pose(
            flat_current, interaction_reference, flat_reference);
    });
    require(
        flat_pose_bits_equal(flat_current, flat_current_before),
        "calibrated expansion mutated invalid current flat input");

    flat_current = make_flat_pose();
    interaction_reference.rotations[25] = quat(0.0F, 0.0F, 0.0F, 0.0F);
    const Pose interaction_reference_before = interaction_reference;
    expect_format_error([&] {
        (void)interaction::expand_flat_controller_pose(
            flat_current, interaction_reference, flat_reference);
    });
    require(
        pose_bits_equal(interaction_reference, interaction_reference_before),
        "calibrated expansion mutated invalid true-G1 reference");

    interaction_reference = make_adversarial_true_g1_reference();
    flat_reference.angular_velocities[14].x =
        std::numeric_limits<float>::infinity();
    const FlatControllerPose flat_reference_before = flat_reference;
    expect_format_error([&] {
        (void)interaction::expand_flat_controller_pose(
            flat_current, interaction_reference, flat_reference);
    });
    require(
        flat_pose_bits_equal(flat_reference, flat_reference_before),
        "calibrated expansion mutated invalid flat reference");
}

void test_reference_retarget_raw_reference_is_exact_flat_reference() {
    const Pose raw_reference = make_pose(0.5F);
    const FlatControllerPose flat_reference = make_flat_pose();
    const FlatControllerPose result = interaction::collapse_interaction_pose(
        raw_reference, raw_reference, flat_reference);
    require(
        flat_pose_bits_equal(result, flat_reference),
        "raw ownership reference did not reproduce displayed flat reference");
}

void test_reference_retarget_fixes_nonroot_translations_under_mismatched_source() {
    const Pose raw_reference = make_pose(0.25F);
    Pose current = raw_reference;
    for (size_t bone = 1; bone < g1_skeleton::BoneCount; ++bone) {
        const float value = static_cast<float>(bone + 1U);
        current.positions[bone] = vec3(
            10.0F * value, -7.0F * value, 13.0F * value);
    }
    const FlatControllerPose flat_reference = make_flat_pose();
    const FlatControllerPose result = interaction::collapse_interaction_pose(
        current, raw_reference, flat_reference);
    for (size_t bone = 1; bone < interaction::kFlatControllerBoneCount;
         ++bone) {
        require(
            vec_bits_equal(
                result.positions[bone], flat_reference.positions[bone]),
            "source morphology changed a target non-root translation");
    }
}

void test_reference_retarget_transfers_mapped_world_rotation_delta() {
    constexpr size_t kFlatSpineUpper = 12U;
    constexpr size_t kG1SpineUpper = 16U;
    const Pose raw_reference = make_pose(0.75F);
    Pose current = raw_reference;
    current.rotations[kG1SpineUpper] = quat_mul(
        quat_from_angle_axis(
            0.35F, normalize(vec3(0.2F, 0.9F, 0.3F))),
        current.rotations[kG1SpineUpper]);
    const FlatControllerPose flat_reference = make_flat_pose();
    const FlatControllerPose result = interaction::collapse_interaction_pose(
        current, raw_reference, flat_reference);
    const interaction::WorldPose source_reference_world =
        interaction::world_pose(raw_reference);
    const interaction::WorldPose source_current_world =
        interaction::world_pose(current);
    const FlatWorldPose target_reference_world =
        flat_world_pose(flat_reference);
    const FlatWorldPose target_current_world = flat_world_pose(result);
    const quat source_world_delta = quat_mul(
        source_current_world.rotations[kG1SpineUpper],
        quat_inv(source_reference_world.rotations[kG1SpineUpper]));
    const quat expected = quat_mul(
        source_world_delta,
        target_reference_world.rotations[kFlatSpineUpper]);
    require_same_rotation(
        target_current_world.rotations[kFlatSpineUpper],
        expected,
        "mapped world-rotation delta was not calibrated onto flat reference");
}

void test_reference_retarget_keeps_neck_and_head_reference_locals() {
    const Pose raw_reference = make_pose(1.0F);
    Pose current = raw_reference;
    current.rotations[16] = quat_mul(
        quat_from_angle_axis(0.08F, vec3(0.0F, 1.0F, 0.0F)),
        current.rotations[16]);
    const FlatControllerPose flat_reference = make_flat_pose();
    const FlatControllerPose result = interaction::collapse_interaction_pose(
        current, raw_reference, flat_reference);
    for (size_t bone : {13U, 14U}) {
        require(vec_bits_equal(result.positions[bone],
                               flat_reference.positions[bone]),
                "Neck/Head local translation left target reference");
        require(quat_bits_equal(result.rotations[bone],
                                flat_reference.rotations[bone]),
                "Neck/Head local rotation left target reference");
    }
    const FlatWorldPose before = flat_world_pose(flat_reference);
    const FlatWorldPose after = flat_world_pose(result);
    require(rotation_distance(before.rotations[13], after.rotations[13]) <
                0.10F,
            "Neck world motion exceeded its continuous parent delta");
    require(rotation_distance(before.rotations[14], after.rotations[14]) <
                0.10F,
            "Head world motion exceeded its continuous parent delta");
}

void test_flat_bridge_accepts_antipodal_equivalent_rotations() {
    const FlatControllerPose positive = make_flat_pose(false);
    const FlatControllerPose antipodal = make_flat_pose(true);
    const Pose reference = make_pose(0.75F, true);
    const Pose positive_expanded =
        interaction::expand_flat_controller_pose(positive, reference);
    const Pose antipodal_expanded =
        interaction::expand_flat_controller_pose(antipodal, reference);
    const FlatWorldPose positive_world = flat_world_pose(positive);
    const FlatWorldPose antipodal_world = flat_world_pose(antipodal);
    const interaction::WorldPose positive_g1 =
        interaction::world_pose(positive_expanded);
    const interaction::WorldPose antipodal_g1 =
        interaction::world_pose(antipodal_expanded);

    for (const interaction::FlatControllerAnchor anchor :
         interaction::kFlatControllerAnchors) {
        assert_same_rotation(
            positive_world.rotations[anchor.flat_bone],
            antipodal_world.rotations[anchor.flat_bone],
            3.0e-4F);
        assert_same_rotation(
            positive_g1.rotations[anchor.g1_bone],
            antipodal_g1.rotations[anchor.g1_bone],
            3.0e-4F);
    }
    assert_flat_pose_near(
        interaction::collapse_interaction_pose(
            antipodal_expanded, positive_expanded, antipodal),
        antipodal);
}

void test_flat_bridge_rejects_invalid_input_without_mutation() {
    const auto expect_format_error = [](const auto& operation) {
        bool threw = false;
        try {
            operation();
        } catch (const interaction::FormatError&) {
            threw = true;
        }
        assert(threw);
    };

    FlatControllerPose invalid_flat = make_flat_pose();
    Pose reference = make_pose(1.0F);
    invalid_flat.positions[4].x =
        std::numeric_limits<float>::quiet_NaN();
    const FlatControllerPose invalid_flat_before = invalid_flat;
    const Pose reference_before = reference;
    expect_format_error([&] {
        (void)interaction::expand_flat_controller_pose(
            invalid_flat, reference);
    });
    assert(flat_pose_bits_equal(invalid_flat, invalid_flat_before));
    assert(pose_bits_equal(reference, reference_before));

    invalid_flat = make_flat_pose();
    invalid_flat.rotations[7] = quat(0.0F, 0.0F, 0.0F, 0.0F);
    const FlatControllerPose zero_rotation_before = invalid_flat;
    expect_format_error([&] {
        (void)interaction::expand_flat_controller_pose(
            invalid_flat, reference);
    });
    assert(flat_pose_bits_equal(invalid_flat, zero_rotation_before));

    const FlatControllerPose valid_flat = make_flat_pose();
    reference = make_pose(1.25F);
    reference.angular_velocities[29].z =
        std::numeric_limits<float>::infinity();
    const Pose invalid_reference_before = reference;
    expect_format_error([&] {
        (void)interaction::expand_flat_controller_pose(valid_flat, reference);
    });
    assert(pose_bits_equal(reference, invalid_reference_before));

    Pose invalid_interaction = make_pose(1.5F);
    invalid_interaction.rotations[12] = quat(0.0F, 0.0F, 0.0F, 0.0F);
    FlatControllerPose fallback = make_flat_pose();
    const Pose invalid_interaction_before = invalid_interaction;
    const FlatControllerPose fallback_before = fallback;
    const Pose collapse_reference = make_pose(1.5F);
    expect_format_error([&] {
        (void)interaction::collapse_interaction_pose(
            invalid_interaction, collapse_reference, fallback);
    });
    assert(pose_bits_equal(invalid_interaction, invalid_interaction_before));
    assert(flat_pose_bits_equal(fallback, fallback_before));

    invalid_interaction = make_pose(1.75F);
    fallback.velocities[18].y =
        std::numeric_limits<float>::quiet_NaN();
    const FlatControllerPose invalid_fallback_before = fallback;
    expect_format_error([&] {
        (void)interaction::collapse_interaction_pose(
            invalid_interaction, collapse_reference, fallback);
    });
    assert(flat_pose_bits_equal(fallback, invalid_fallback_before));
}

void test_flat_bridge_canaries_prove_exact_23_element_bounds() {
    struct GuardedFlatPose {
        std::array<uint64_t, 4> before{};
        FlatControllerPose pose{};
        std::array<uint64_t, 4> after{};
    };
    constexpr std::array<uint64_t, 4> kBefore = {
        0x0123456789abcdefULL,
        0xfedcba9876543210ULL,
        0xaaaaaaaa55555555ULL,
        0x13579bdf2468ace0ULL};
    constexpr std::array<uint64_t, 4> kAfter = {
        0x0f0e0d0c0b0a0908ULL,
        0x1020304050607080ULL,
        0xcafebabedeadbeefULL,
        0x55aa55aa33cc33ccULL};

    GuardedFlatPose input{kBefore, make_flat_pose(), kAfter};
    GuardedFlatPose output{kBefore, {}, kAfter};
    const Pose reference = make_pose(2.0F);
    const Pose expanded = interaction::expand_flat_controller_pose(
        input.pose, reference);
    output.pose = interaction::collapse_interaction_pose(
        expanded, expanded, input.pose);
    assert(input.before == kBefore);
    assert(input.after == kAfter);
    assert(output.before == kBefore);
    assert(output.after == kAfter);
    assert_flat_pose_near(output.pose, input.pose);
}

void test_scheduler_cadence_and_cache() {
    ControllerInteractionScheduler scheduler;
    assert(scheduler.phase() == 0);
    assert(!scheduler.updated_last_tick());

    int snapshot_calls = 0;
    int resolver_calls = 0;
    int update_calls = 0;
    std::vector<int> due_ticks;
    RuntimeOutput newest{};

    for (int tick = 1; tick <= 25; ++tick) {
        const int calls_before = update_calls;
        const RuntimeOutput before = scheduler.cached_output();
        const RuntimeOutput& observed = scheduler.tick(
            {},
            [&]() {
                ++snapshot_calls;
                return make_snapshot(static_cast<float>(tick));
            },
            [&](const LocomotionSnapshot&) -> std::optional<PickRequest> {
                ++resolver_calls;
                return std::nullopt;
            },
            [&](const RuntimeInput& input) {
                ++update_calls;
                due_ticks.push_back(tick);
                assert(input.dt == 1.0F / 25.0F);
                assert(!input.interact_pressed);
                assert(!input.pick_request.has_value());
                newest = make_complete_output(
                    update_calls,
                    (update_calls % 2) == 0
                        ? RuntimeState::Disabled
                        : RuntimeState::Locomotion);
                return newest;
            });

        assert(update_calls == calls_before + 1);
        assert(scheduler.updated_last_tick());
        assert(output_fields_equal(observed, newest));
        assert(!output_fields_equal(observed, before));
        assert(scheduler.phase() == 0);
    }

    assert(update_calls == 25);
    assert(snapshot_calls == 25);
    assert(resolver_calls == 0);
    assert(scheduler.phase() == 0);
    for (int tick = 1; tick <= 25; ++tick) {
        assert(due_ticks.at(static_cast<size_t>(tick - 1)) == tick);
    }
}

void test_edges_latch_coalesce_and_clear_after_delivery() {
    ControllerInteractionScheduler scheduler;
    int snapshot_calls = 0;
    int resolver_calls = 0;
    int update_calls = 0;
    std::vector<RuntimeInput> delivered;
    const PickRequest request{{42, 7}, 9, 1234};

    const auto snapshot_provider = [&]() {
        ++snapshot_calls;
        return make_snapshot(4.0F);
    };
    const auto resolver = [&](const LocomotionSnapshot& snapshot) {
        ++resolver_calls;
        assert(pose_bits_equal(snapshot.pose, make_pose(4.0F)));
        return std::optional<PickRequest>{request};
    };
    const auto update = [&](const RuntimeInput& input) {
        ++update_calls;
        delivered.push_back(input);
        return make_complete_output(update_calls, RuntimeState::Locomotion);
    };

    scheduler.tick({true, false, false}, snapshot_provider, resolver, update);
    assert(snapshot_calls == 1);
    assert(resolver_calls == 1);
    assert(update_calls == 1);
    assert(delivered[0].interact_pressed);
    assert(!delivered[0].cancel_pressed);
    assert(!delivered[0].reset_pressed);
    assert(delivered[0].pick_request.has_value());
    assert(delivered[0].pick_request->target == request.target);
    assert(delivered[0].pick_request->affordance_id == request.affordance_id);
    assert(delivered[0].pick_request->request_id == request.request_id);

    scheduler.tick({false, true, true}, snapshot_provider, resolver, update);
    assert(snapshot_calls == 2);
    assert(resolver_calls == 1);
    assert(update_calls == 2);
    assert(!delivered[1].interact_pressed);
    assert(delivered[1].cancel_pressed);
    assert(delivered[1].reset_pressed);
    assert(!delivered[1].pick_request.has_value());

    scheduler.tick(
        {true, true, true}, snapshot_provider, resolver, update);
    const RuntimeInput& immediate = delivered.back();
    assert(immediate.interact_pressed);
    assert(immediate.cancel_pressed);
    assert(immediate.reset_pressed);
    assert(update_calls == 3);
    assert(scheduler.phase() == 0);
}

void test_manual_pick_request_reset_priority_discards_stale_latch() {
    ControllerInteractionScheduler scheduler;
    std::optional<PickRequest> manual_request;
    uint64_t next_request_id = 71U;
    uint32_t resolver_calls = 0U;
    uint32_t exchanges = 0U;
    bool persistent_manual_post_output = true;
    std::vector<RuntimeInput> delivered;
    const PickRequest request{{42U, 7U}, 9U, 71U};

    const auto snapshot = [] { return make_snapshot(4.0F); };
    const auto resolve_pick = [&](const LocomotionSnapshot&)
        -> std::optional<PickRequest> {
        ++resolver_calls;
        if (!manual_request.has_value()) {
            return std::nullopt;
        }
        ++exchanges;
        std::optional<PickRequest> submission =
            std::exchange(manual_request, std::nullopt);
        require(
            submission->request_id == next_request_id,
            "resolver accepted a request with the wrong pre-increment ID");
        ++next_request_id;
        return submission;
    };
    const auto no_place = [](const LocomotionSnapshot&)
        -> std::optional<ControllerPlaceTarget> { return std::nullopt; };
    const auto no_preview = [](SurfaceHandle, uint32_t) {
        return PlaceStagingPreview{};
    };
    const auto update = [&](const RuntimeInput& input) {
        delivered.push_back(input);
        return make_complete_output(
            static_cast<int>(delivered.size()), RuntimeState::Locomotion);
    };

    manual_request = request;
    (void)scheduler.tick(
        {true, false, true},
        snapshot,
        resolve_pick,
        no_place,
        no_preview,
        update);
    manual_request.reset();
    if (delivered.back().reset_pressed) {
        persistent_manual_post_output = false;
    }
    require(
        resolver_calls == 0U && exchanges == 0U &&
            next_request_id == 71U && !manual_request.has_value() &&
            !persistent_manual_post_output &&
            delivered.back().reset_pressed &&
            !delivered.back().interact_pressed &&
            !delivered.back().pick_request.has_value(),
        "reset priority consumed or retained the same-tick manual latch");

    (void)scheduler.tick(
        {true, false, false},
        snapshot,
        resolve_pick,
        no_place,
        no_preview,
        update);
    manual_request.reset();
    require(
        resolver_calls == 1U && exchanges == 0U &&
            next_request_id == 71U &&
            !delivered.back().pick_request.has_value(),
        "a later F submitted the request skipped by reset priority");

    manual_request = request;
    (void)scheduler.tick(
        {true, false, false},
        snapshot,
        resolve_pick,
        no_place,
        no_preview,
        update);
    manual_request.reset();
    require(
        resolver_calls == 2U && exchanges == 1U &&
            next_request_id == 72U && !manual_request.has_value() &&
            delivered.back().pick_request.has_value() &&
            delivered.back().pick_request->request_id == 71U,
        "accepted manual request did not exchange once with its prior ID");
}

void test_cache_changes_only_after_successful_due_delivery() {
    ControllerInteractionScheduler scheduler;
    assert(!scheduler.updated_last_tick());
    const auto snapshot_provider = [] { return make_snapshot(2.0F); };
    const auto resolver = [](const LocomotionSnapshot&) {
        return std::optional<PickRequest>{PickRequest{{8, 2}, 5, 99}};
    };
    int calls = 0;

    const RuntimeOutput initial{};
    assert(output_fields_equal(scheduler.cached_output(), initial));

    bool threw = false;
    try {
        scheduler.tick({true, false, false}, snapshot_provider, resolver,
                       [&](const RuntimeInput& input) -> RuntimeOutput {
                           ++calls;
                           assert(input.interact_pressed);
                           throw std::runtime_error("delivery failed");
                       });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(calls == 1);
    assert(!scheduler.updated_last_tick());
    assert(output_fields_equal(scheduler.cached_output(), initial));
    assert(scheduler.phase() == 0);

    scheduler.tick({}, snapshot_provider, resolver,
                   [&](const RuntimeInput& input) {
                       ++calls;
                       assert(input.interact_pressed);
                       return make_complete_output(2, RuntimeState::Locomotion);
                   });
    assert(calls == 2);
    assert(scheduler.updated_last_tick());
    assert(!output_fields_equal(scheduler.cached_output(), initial));
    assert(scheduler.phase() == 0);
}

void test_scheduler_latches_carry_place_and_submits_only_live_ready_preview() {
    ControllerInteractionScheduler scheduler;
    const ControllerPlaceTarget destination{
        SurfaceHandle{901U, 4U}, 77U, 3001U};
    const TargetHandle held_target{42U, 8U};
    int pick_resolver_calls = 0;
    int place_resolver_calls = 0;
    int preview_calls = 0;
    std::vector<RuntimeInput> delivered;

    auto snapshot_provider = [] {
        LocomotionSnapshot snapshot = make_snapshot(2.0F);
        snapshot.pose.positions[0] = vec3(1.0F, 0.0F, 2.0F);
        return snapshot;
    };
    auto pick_resolver = [&](const LocomotionSnapshot&)
        -> std::optional<PickRequest> {
        ++pick_resolver_calls;
        return std::nullopt;
    };
    auto place_resolver = [&](const LocomotionSnapshot& snapshot)
        -> std::optional<ControllerPlaceTarget> {
        ++place_resolver_calls;
        require_vec_near(
            snapshot.pose.positions[0],
            vec3(1.0F, 0.0F, 2.0F),
            "manual place resolver received the wrong live root");
        return destination;
    };
    auto preview_resolver = [&](SurfaceHandle surface, uint32_t affordance_id) {
        ++preview_calls;
        require(
            surface == destination.surface &&
                affordance_id == destination.affordance_id,
            "place preview did not receive the exact latched identity");
        PlaceStagingPreview preview{};
        preview.accepted = true;
        preview.ready = preview_calls >= 2;
        preview.root_error_m = preview.ready ? 0.20F : 0.80F;
        preview.yaw_error_radians = preview.ready ? 0.20F : 0.60F;
        preview.candidate.selection_id = preview.ready ? 222U : 111U;
        preview.staging_root_world = {
            vec3(1.25F, 0.0F, 2.10F),
            quat_from_angle_axis(0.20F, vec3(0.0F, 1.0F, 0.0F))};
        return preview;
    };
    auto runtime_update = [&](const RuntimeInput& input) {
        delivered.push_back(input);
        RuntimeOutput output;
        output.owns_pose = true;
        output.diagnostics.state = delivered.size() >= 3U
            ? RuntimeState::PlacePreflight
            : RuntimeState::Carry;
        output.diagnostics.target = held_target;
        output.diagnostics.object_state = ObjectState::Held;
        output.diagnostics.attached = true;
        return output;
    };

    (void)scheduler.tick(
        {}, snapshot_provider, pick_resolver, place_resolver,
        preview_resolver, runtime_update);
    require(
        delivered.size() == 1U &&
            delivered.back().interact_pressed == false,
        "Carry setup unexpectedly pulsed Interact");

    (void)scheduler.tick(
        {true, false, false},
        snapshot_provider,
        pick_resolver,
        place_resolver,
        preview_resolver,
        runtime_update);
    require(place_resolver_calls == 1, "Carry F did not resolve one surface");
    require(pick_resolver_calls == 0, "Carry F invoked the pick resolver");
    require(preview_calls == 1, "Carry F did not compute its first preview");
    require(
        !delivered.back().interact_pressed &&
            !delivered.back().place_request.has_value(),
        "far Carry preview immediately pulsed runtime Place");
    require(
        scheduler.latched_place().has_value() &&
            scheduler.latched_place()->surface == destination.surface &&
            scheduler.latched_place()->affordance_id ==
                destination.affordance_id,
        "Carry F did not retain the exact destination identity");
    require(
        scheduler.place_preview().has_value() &&
            scheduler.place_preview()->accepted &&
            !scheduler.place_preview()->ready &&
            scheduler.place_preview()->candidate.selection_id == 111U,
        "far accepted preview lost its staging candidate");

    (void)scheduler.tick(
        {}, snapshot_provider, pick_resolver, place_resolver,
        preview_resolver, runtime_update);
    require(preview_calls == 2, "latched place was not previewed every tick");
    require(
        delivered.back().interact_pressed &&
            delivered.back().place_request.has_value(),
        "ready live preview did not pulse runtime Place once");
    const interaction::PlaceRequest& request =
        *delivered.back().place_request;
    require(
        request.held_target == held_target &&
            request.surface == destination.surface &&
            request.affordance_id == destination.affordance_id &&
            request.request_id == destination.request_id &&
            request.selection_id == 222U,
        "Place request did not use the newly recomputed live preview");
    require(
        request.selection_id != 111U,
        "Place request submitted the prior far preview selection");
}

void test_scheduler_cancel_clears_place_latch_before_another_preview() {
    ControllerInteractionScheduler scheduler;
    const ControllerPlaceTarget destination{
        SurfaceHandle{902U, 5U}, 88U, 4001U};
    int preview_calls = 0;
    RuntimeOutput carry;
    carry.owns_pose = true;
    carry.diagnostics.state = RuntimeState::Carry;
    carry.diagnostics.target = {52U, 2U};
    carry.diagnostics.object_state = ObjectState::Held;
    carry.diagnostics.attached = true;
    auto update = [&](const RuntimeInput& input) {
        if (input.cancel_pressed) {
            require(
                !input.interact_pressed &&
                    !input.place_request.has_value(),
                "Cancel delivered a Place request");
        }
        return carry;
    };
    auto no_pick = [](const LocomotionSnapshot&)
        -> std::optional<PickRequest> { return std::nullopt; };
    auto resolve_place = [&](const LocomotionSnapshot&)
        -> std::optional<ControllerPlaceTarget> { return destination; };
    auto preview = [&](SurfaceHandle, uint32_t) {
        ++preview_calls;
        PlaceStagingPreview result{};
        result.accepted = true;
        result.root_error_m = 1.0F;
        result.yaw_error_radians = 1.0F;
        result.candidate.selection_id = 123U;
        return result;
    };

    (void)scheduler.tick(
        {}, [] { return LocomotionSnapshot{}; }, no_pick,
        resolve_place, preview, update);
    (void)scheduler.tick(
        {true, false, false},
        [] { return LocomotionSnapshot{}; },
        no_pick,
        resolve_place,
        preview,
        update);
    require(
        scheduler.latched_place().has_value() && preview_calls == 1,
        "place setup did not latch one preview");

    (void)scheduler.tick(
        {true, true, false},
        [] { return LocomotionSnapshot{}; },
        no_pick,
        resolve_place,
        preview,
        update);
    require(
        !scheduler.latched_place().has_value() &&
            !scheduler.place_preview().has_value(),
        "X did not clear the latched place target and preview");
    require(
        preview_calls == 1,
        "X recomputed a preview after clearing its latch");
}

void test_scheduler_reset_clears_place_latch_before_another_preview() {
    ControllerInteractionScheduler scheduler;
    const ControllerPlaceTarget destination{
        SurfaceHandle{903U, 6U}, 89U, 4002U};
    int preview_calls = 0;
    RuntimeOutput carry;
    carry.owns_pose = true;
    carry.diagnostics.state = RuntimeState::Carry;
    carry.diagnostics.target = {53U, 3U};
    carry.diagnostics.object_state = ObjectState::Held;
    carry.diagnostics.attached = true;
    auto update = [&](const RuntimeInput& input) {
        if (input.reset_pressed) {
            require(
                !input.interact_pressed &&
                    !input.place_request.has_value(),
                "Reset delivered a Place request");
        }
        return carry;
    };
    auto no_pick = [](const LocomotionSnapshot&)
        -> std::optional<PickRequest> { return std::nullopt; };
    auto resolve_place = [&](const LocomotionSnapshot&)
        -> std::optional<ControllerPlaceTarget> { return destination; };
    auto preview = [&](SurfaceHandle, uint32_t) {
        ++preview_calls;
        PlaceStagingPreview result{};
        result.accepted = true;
        result.root_error_m = 1.0F;
        result.yaw_error_radians = 1.0F;
        result.candidate.selection_id = 124U;
        return result;
    };

    (void)scheduler.tick(
        {}, [] { return LocomotionSnapshot{}; }, no_pick,
        resolve_place, preview, update);
    (void)scheduler.tick(
        {true, false, false},
        [] { return LocomotionSnapshot{}; },
        no_pick,
        resolve_place,
        preview,
        update);
    require(
        scheduler.latched_place().has_value() && preview_calls == 1,
        "place setup did not latch one preview before reset");

    (void)scheduler.tick(
        {true, false, true},
        [] { return LocomotionSnapshot{}; },
        no_pick,
        resolve_place,
        preview,
        update);
    require(
        !scheduler.latched_place().has_value() &&
            !scheduler.place_preview().has_value(),
        "R did not clear the latched place target and preview");
    require(
        preview_calls == 1,
        "R recomputed a preview after clearing its latch");
}

void test_scheduler_clears_far_place_when_carry_authority_ends() {
    ControllerInteractionScheduler scheduler;
    const ControllerPlaceTarget destination{
        SurfaceHandle{904U, 7U}, 90U, 4003U};
    int place_resolver_calls = 0;
    int preview_calls = 0;
    std::vector<RuntimeInput> delivered;

    RuntimeOutput carry;
    carry.owns_pose = true;
    carry.diagnostics.state = RuntimeState::Carry;
    carry.diagnostics.target = {54U, 4U};
    carry.diagnostics.object_state = ObjectState::Held;
    carry.diagnostics.attached = true;
    RuntimeOutput next_output = carry;

    auto no_pick = [](const LocomotionSnapshot&)
        -> std::optional<PickRequest> { return std::nullopt; };
    auto resolve_place = [&](const LocomotionSnapshot&)
        -> std::optional<ControllerPlaceTarget> {
        ++place_resolver_calls;
        return destination;
    };
    auto preview = [&](SurfaceHandle surface, uint32_t affordance_id) {
        ++preview_calls;
        require(
            surface == destination.surface &&
                affordance_id == destination.affordance_id,
            "stale-place test previewed the wrong destination");
        PlaceStagingPreview result{};
        result.accepted = true;
        result.ready = false;
        result.root_error_m = 0.80F;
        result.yaw_error_radians = 0.60F;
        result.candidate.selection_id = 125U;
        return result;
    };
    auto update = [&](const RuntimeInput& input) {
        delivered.push_back(input);
        return next_output;
    };
    auto snapshot = [] { return LocomotionSnapshot{}; };

    (void)scheduler.tick(
        {}, snapshot, no_pick, resolve_place, preview, update);
    (void)scheduler.tick(
        {true, false, false},
        snapshot,
        no_pick,
        resolve_place,
        preview,
        update);
    require(
        scheduler.latched_place().has_value() &&
            scheduler.place_preview().has_value() &&
            preview_calls == 1 && place_resolver_calls == 1,
        "far Carry setup did not retain one manual placement preview");

    RuntimeOutput locomotion;
    locomotion.diagnostics.state = RuntimeState::Locomotion;
    locomotion.diagnostics.target = {54U, 5U};
    locomotion.diagnostics.object_state = ObjectState::Free;
    next_output = locomotion;
    (void)scheduler.tick(
        {}, snapshot, no_pick, resolve_place, preview, update);
    require(
        !scheduler.latched_place().has_value() &&
            !scheduler.place_preview().has_value(),
        "external Carry authority loss retained far placement staging");
    require(
        !delivered.back().interact_pressed &&
            !delivered.back().place_request.has_value(),
        "external Carry authority loss submitted the far placement");
    const int previews_after_authority_loss = preview_calls;

    carry.diagnostics.target = {54U, 6U};
    next_output = carry;
    (void)scheduler.tick(
        {}, snapshot, no_pick, resolve_place, preview, update);
    (void)scheduler.tick(
        {}, snapshot, no_pick, resolve_place, preview, update);
    require(
        preview_calls == previews_after_authority_loss &&
            place_resolver_calls == 1,
        "later Carry previewed the stale destination without a fresh F");
    require(
        !delivered.back().interact_pressed &&
            !delivered.back().place_request.has_value(),
        "later Carry submitted the stale destination without a fresh F");

    (void)scheduler.tick(
        {true, false, false},
        snapshot,
        no_pick,
        resolve_place,
        preview,
        update);
    require(
        place_resolver_calls == 2 &&
            preview_calls == previews_after_authority_loss + 1,
        "fresh manual F did not start a new placement staging epoch");
}

void require_flat_pose_near(
    const FlatControllerPose& actual,
    const FlatControllerPose& expected,
    const char* message,
    float tolerance = 3.0e-4F) {
    for (size_t bone = 0; bone < actual.positions.size(); ++bone) {
        require_vec_near(
            actual.positions[bone], expected.positions[bone], message, tolerance);
        require_vec_near(
            actual.velocities[bone], expected.velocities[bone], message, tolerance);
        require_same_rotation(
            actual.rotations[bone], expected.rotations[bone], message, tolerance);
        require_vec_near(
            actual.angular_velocities[bone],
            expected.angular_velocities[bone],
            message,
            tolerance);
    }
    require(actual.foot_contacts == expected.foot_contacts, message);
}

RuntimeOutput make_owned_output(const Pose& raw_pose) {
    RuntimeOutput output;
    output.owns_pose = true;
    output.pose = raw_pose;
    output.diagnostics.state = RuntimeState::Align;
    return output;
}

ControllerInteractionHandConstraint make_hand_constraint(
    const RuntimeOutput& output,
    const FlatControllerPose& base_pose,
    Hand hand) {
    const FlatWorldPose world = flat_world_pose(base_pose);
    const size_t upper_arm_bone = hand == Hand::Left ? 16U : 20U;
    const size_t hand_bone = hand == Hand::Left ? 18U : 22U;
    ControllerInteractionHandConstraint constraint;
    constraint.target = output.diagnostics.target;
    constraint.affordance_id = output.diagnostics.affordance_id;
    constraint.hand = hand;
    constraint.grasp_world = {
        lerp(
            world.positions[upper_arm_bone],
            world.positions[hand_bone],
            0.80F) + vec3(0.0F, 0.02F, -0.01F),
        world.rotations[hand_bone]};
    return constraint;
}

float rotation_distance(quat left, quat right);

void test_frame_handoff_first_owned_frame_is_exact_displayed_reference() {
    const FlatControllerPose entry = make_flat_pose();
    RuntimeOutput owned = make_owned_output(make_pose(0.375F));
    ControllerInteractionFrameHandoff handoff;
    const ControllerInteractionFrameState frame = handoff.apply(
        entry, owned, interaction::kControllerStepSeconds);
    require(frame.runtime_owns_pose, "owned frame lost runtime ownership");
    require(frame.overrides_locomotion_pose, "owned frame lost visual override");
    require(
        flat_pose_bits_equal(frame.pose, entry),
        "first ownership frame did not preserve displayed entry pose");
}

void test_frame_handoff_solves_positive_targeted_constraint_on_owned_entry() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput owned = make_owned_output(raw_reference);
    owned.diagnostics.target = {70U, 3U};
    owned.diagnostics.affordance_id = 8U;
    owned.diagnostics.hand = Hand::Right;
    owned.diagnostics.object_state = ObjectState::Targeted;
    owned.diagnostics.attached = false;
    owned.diagnostics.hand_constraint_weight = 0.04F;
    const ControllerInteractionHandConstraint constraint =
        make_hand_constraint(owned, entry, Hand::Right);

    ControllerInteractionFrameHandoff handoff;
    const ControllerInteractionFrameState targeted = handoff.apply(
        entry,
        owned,
        interaction::kControllerStepSeconds,
        constraint);
    require(
        targeted.runtime_owns_pose &&
            targeted.hand_constraint_validated &&
            targeted.hand_constraint_result.applied &&
            targeted.hand_constraint_result.reachable,
        "positive-weight owned Targeted entry did not publish a validated "
        "reachable hand solve");
}

void test_calibrated_bridge_keeps_exact_25_hz_handoff_continuous_and_reaches_oracle() {
    FlatControllerPose entry = make_flat_pose();
    for (size_t bone = 1; bone < entry.positions.size(); ++bone) {
        entry.positions[bone] = 3.0F * entry.positions[bone];
    }
    FlatControllerPose canonical_oracle = entry;
    canonical_oracle.positions[0] =
        canonical_oracle.positions[0] + vec3(0.05F, 0.0F, -0.03F);
    canonical_oracle.velocities[0] =
        canonical_oracle.velocities[0] + vec3(0.02F, 0.0F, -0.01F);

    Pose true_g1_reference = make_adversarial_true_g1_reference();
    true_g1_reference.foot_contacts = entry.foot_contacts;
    const Pose calibrated_live = interaction::expand_flat_controller_pose(
        entry, true_g1_reference, entry);
    const Pose canonical_source = interaction::expand_flat_controller_pose(
        canonical_oracle, true_g1_reference, entry);
    const Pose legacy_absolute_anchor =
        interaction::expand_flat_controller_pose(entry, true_g1_reference);

    ControllerInteractionFrameHandoff calibrated_handoff;
    ControllerInteractionFrameHandoff legacy_handoff;
    RuntimeOutput calibrated_output = make_owned_output(calibrated_live);
    RuntimeOutput legacy_output = make_owned_output(legacy_absolute_anchor);
    ControllerInteractionFrameState calibrated_frame = calibrated_handoff.apply(
        entry,
        calibrated_output,
        interaction::kControllerStepSeconds);
    ControllerInteractionFrameState legacy_frame = legacy_handoff.apply(
        entry,
        legacy_output,
        interaction::kControllerStepSeconds);

    float calibrated_max_translation = 0.0F;
    float calibrated_max_rotation = 0.0F;
    float legacy_max_translation = 0.0F;
    constexpr float kEntryBlendSeconds = 0.25F;
    for (int32_t tick = 1; tick <= 7; ++tick) {
        const float elapsed_seconds =
            static_cast<float>(tick) * interaction::kControllerStepSeconds;
        const float alpha = std::clamp(
            elapsed_seconds / kEntryBlendSeconds, 0.0F, 1.0F);
        calibrated_output.pose = interaction::interpolate_pose(
            calibrated_live, canonical_source, alpha);
        legacy_output.pose = interaction::interpolate_pose(
            legacy_absolute_anchor, canonical_source, alpha);

        const ControllerInteractionFrameState next_calibrated =
            calibrated_handoff.apply(
                entry,
                calibrated_output,
                interaction::kControllerStepSeconds);
        const ControllerInteractionFrameState next_legacy = legacy_handoff.apply(
            entry,
            legacy_output,
            interaction::kControllerStepSeconds);
        const FlatWorldPose calibrated_before = flat_world_pose(
            calibrated_frame.pose);
        const FlatWorldPose calibrated_after = flat_world_pose(
            next_calibrated.pose);
        const FlatWorldPose legacy_before = flat_world_pose(legacy_frame.pose);
        const FlatWorldPose legacy_after = flat_world_pose(next_legacy.pose);
        for (size_t bone = 0; bone < entry.positions.size(); ++bone) {
            calibrated_max_translation = std::max(
                calibrated_max_translation,
                length(
                    calibrated_after.positions[bone] -
                    calibrated_before.positions[bone]));
            calibrated_max_rotation = std::max(
                calibrated_max_rotation,
                rotation_distance(
                    calibrated_before.rotations[bone],
                    calibrated_after.rotations[bone]));
            legacy_max_translation = std::max(
                legacy_max_translation,
                length(
                    legacy_after.positions[bone] -
                    legacy_before.positions[bone]));
        }
        calibrated_frame = next_calibrated;
        legacy_frame = next_legacy;
    }

    require(
        legacy_max_translation > 0.20F,
        "legacy absolute-anchor bridge did not reproduce the >0.20 m basis jump");
    require(
        calibrated_max_translation <= 0.20F + 1.0e-5F,
        "calibrated bridge exceeded the 0.20 m exact-25-Hz step limit");
    require(
        calibrated_max_rotation <= 1.047197551F + 1.0e-5F,
        "calibrated bridge exceeded the 60-degree exact-25-Hz step limit");
    require_flat_pose_near(
        calibrated_frame.pose,
        canonical_oracle,
        "calibrated bridge merely smoothed and did not reach the oracle",
        7.0e-4F);
}

void test_frame_handoff_applies_validated_semantic_hand_constraint_and_releases_from_it() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput owned = make_owned_output(raw_reference);
    owned.diagnostics.target = {71U, 4U};
    owned.diagnostics.affordance_id = 9U;
    owned.diagnostics.hand = Hand::Right;
    owned.diagnostics.state = RuntimeState::Hold;
    owned.diagnostics.object_state = ObjectState::Held;
    owned.diagnostics.attached = true;
    owned.diagnostics.hand_constraint_weight = 0.0F;

    const FlatWorldPose entry_world = flat_world_pose(entry);
    ControllerInteractionHandConstraint constraint;
    constraint.target = owned.diagnostics.target;
    constraint.affordance_id = owned.diagnostics.affordance_id;
    constraint.hand = owned.diagnostics.hand;
    constraint.grasp_world = {
        entry_world.positions[22] + vec3(0.03F, 0.02F, -0.01F),
        entry_world.rotations[22]};
    const ControllerInteractionHandConstraint constraint_before = constraint;
    ControllerInteractionFrameHandoff handoff;
    const ControllerInteractionFrameState first = handoff.apply(
        entry,
        owned,
        interaction::kControllerStepSeconds,
        constraint);
    interaction::TargetRigArmIK expected_solver;
    expected_solver.begin_epoch(raw_reference, entry, Hand::Right);
    require(
        first.hand_constraint_validated,
        "matching ownership entry did not validate its exact constraint");
    require_same_rotation(
        first.hand_constraint_calibration_rotation,
        expected_solver.calibration_rotation(),
        "ownership entry did not expose the active solver calibration");
    require(
        flat_pose_bits_equal(first.pose, entry),
        "zero-weight ownership entry changed the displayed pose");
    require(
        !first.hand_constraint_result.applied,
        "zero-weight ownership entry applied the hand constraint");

    owned.diagnostics.hand_constraint_weight = 1.0F;
    const RuntimeOutput constrained_input_before = owned;
    const ControllerInteractionFrameState constrained = handoff.apply(
        entry,
        owned,
        interaction::kControllerStepSeconds,
        constraint);
    require(
        constrained.hand_constraint_result.applied &&
            constrained.hand_constraint_result.reachable,
        "matching full-weight semantic constraint did not solve");
    require(
        constrained.hand_constraint_validated,
        "matching full-weight constraint lost validation");
    require_same_rotation(
        constrained.hand_constraint_calibration_rotation,
        first.hand_constraint_calibration_rotation,
        "active epoch calibration changed between rendered frames");
    const FlatWorldPose constrained_world = flat_world_pose(constrained.pose);
    require_vec_near(
        constrained_world.positions[22],
        constraint.grasp_world.position,
        "selected hand did not reach the semantic grasp",
        3.0e-4F);
    for (size_t bone = 0; bone < entry.positions.size(); ++bone) {
        const bool selected_chain =
            bone == 19U || bone == 20U || bone == 21U || bone == 22U;
        if (selected_chain) continue;
        require(
            vec_bits_equal(
                constrained.pose.positions[bone], entry.positions[bone]) &&
                vec_bits_equal(
                    constrained.pose.velocities[bone], entry.velocities[bone]) &&
                quat_bits_equal(
                    constrained.pose.rotations[bone], entry.rotations[bone]) &&
                vec_bits_equal(
                    constrained.pose.angular_velocities[bone],
                    entry.angular_velocities[bone]),
            "hand constraint modified a bone outside the selected chain");
    }
    require(
        constrained.pose.foot_contacts == entry.foot_contacts,
        "hand constraint modified foot contacts");
    require(
        output_fields_equal(owned, constrained_input_before) &&
            constraint.target == constraint_before.target &&
            constraint.affordance_id == constraint_before.affordance_id &&
            constraint.hand == constraint_before.hand &&
            transform_bits_equal(
                constraint.grasp_world, constraint_before.grasp_world),
        "hand constraint mutated runtime or scene input data");

    RuntimeOutput idle;
    const ControllerInteractionFrameState first_release = handoff.apply(
        entry,
        idle,
        interaction::kControllerStepSeconds,
        std::nullopt);
    require(
        flat_pose_bits_equal(first_release.pose, constrained.pose),
        "release did not begin at the last constrained pose");

    float release_elapsed_seconds = 0.0F;
    while (release_elapsed_seconds + interaction::kControllerStepSeconds <
           0.25F) {
        release_elapsed_seconds += interaction::kControllerStepSeconds;
        const ControllerInteractionFrameState release = handoff.apply(
            entry,
            idle,
            interaction::kControllerStepSeconds,
            std::nullopt);
        require(
            release.overrides_locomotion_pose,
            "normal 0.25 second release ended early");
    }
    ControllerInteractionFrameState relinquished{};
    do {
        release_elapsed_seconds += interaction::kControllerStepSeconds;
        relinquished = handoff.apply(
            entry,
            idle,
            interaction::kControllerStepSeconds,
            std::nullopt);
    } while (relinquished.overrides_locomotion_pose &&
             release_elapsed_seconds <=
                 0.25F + interaction::kControllerStepSeconds);
    require(
        !relinquished.overrides_locomotion_pose &&
            flat_pose_bits_equal(relinquished.pose, entry),
        "normal 0.25 second release did not relinquish on time");
    require(
        release_elapsed_seconds >= 0.25F &&
            release_elapsed_seconds <=
                0.25F + interaction::kControllerStepSeconds,
        "normal release duration escaped one 25 Hz tick of 0.25 seconds");
}

void test_frame_handoff_starts_matching_constraint_after_attachment_inside_owned_epoch() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput owned = make_owned_output(raw_reference);
    owned.diagnostics.target = {72U, 5U};
    owned.diagnostics.affordance_id = 10U;
    owned.diagnostics.hand = Hand::Right;
    owned.diagnostics.object_state = ObjectState::Targeted;
    owned.diagnostics.hand_constraint_weight = 0.0F;

    ControllerInteractionFrameHandoff handoff;
    const ControllerInteractionFrameState targeted = handoff.apply(
        entry,
        owned,
        interaction::kControllerStepSeconds,
        std::nullopt);
    require(
        targeted.runtime_owns_pose &&
            !targeted.hand_constraint_validated &&
            !targeted.hand_constraint_result.applied,
        "Targeted ownership entry unexpectedly started a hand constraint");

    FlatControllerPose later_locomotion = entry;
    later_locomotion.rotations[22] = quat_from_angle_axis(
        0.17F, normalize(vec3(0.3F, 0.8F, 0.4F)));
    FlatControllerPose later_runtime = entry;
    later_runtime.rotations[22] = quat_from_angle_axis(
        -0.12F, normalize(vec3(0.7F, 0.2F, 0.5F)));
    owned.pose = interaction::expand_flat_controller_pose(
        later_runtime, raw_reference);
    owned.diagnostics.state = RuntimeState::PickupReplay;
    owned.diagnostics.object_state = ObjectState::Attached;
    owned.diagnostics.attached = true;
    owned.diagnostics.hand_constraint_weight = 1.0F;
    const ControllerInteractionHandConstraint constraint =
        make_hand_constraint(owned, entry, Hand::Right);
    const ControllerInteractionFrameState attached = handoff.apply(
        later_locomotion,
        owned,
        interaction::kControllerStepSeconds,
        constraint);
    interaction::TargetRigArmIK expected_solver;
    expected_solver.begin_epoch(raw_reference, entry, Hand::Right);
    require(
        attached.hand_constraint_validated &&
            attached.hand_constraint_result.applied &&
            attached.hand_constraint_result.reachable,
        "matching attached constraint did not start inside active pose ownership");
    require_same_rotation(
        attached.hand_constraint_calibration_rotation,
        expected_solver.calibration_rotation(),
        "delayed constraint did not use the stored ownership references");

    owned.diagnostics.state = RuntimeState::Carry;
    owned.diagnostics.object_state = ObjectState::Held;
    owned.diagnostics.recorded_carry = true;
    const ControllerInteractionFrameState held = handoff.apply(
        later_locomotion,
        owned,
        interaction::kControllerStepSeconds,
        constraint);
    require(
        held.hand_constraint_validated &&
            held.hand_constraint_result.applied &&
            held.hand_constraint_result.reachable,
        "active delayed constraint did not remain solved in Carry");
    require_same_rotation(
        held.hand_constraint_calibration_rotation,
        attached.hand_constraint_calibration_rotation,
        "active delayed constraint restarted after entering Carry");
}

void test_delayed_constraint_rejects_identity_switched_before_first_constraint() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput owned = make_owned_output(raw_reference);
    owned.diagnostics.target = {721U, 5U};
    owned.diagnostics.affordance_id = 101U;
    owned.diagnostics.hand = Hand::Right;
    owned.diagnostics.object_state = ObjectState::Targeted;

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        entry,
        owned,
        interaction::kControllerStepSeconds,
        std::nullopt);

    RuntimeOutput switched = owned;
    switched.diagnostics.state = RuntimeState::PickupReplay;
    switched.diagnostics.target = {722U, 6U};
    switched.diagnostics.affordance_id = 102U;
    switched.diagnostics.hand = Hand::Left;
    switched.diagnostics.object_state = ObjectState::Attached;
    switched.diagnostics.attached = true;
    switched.diagnostics.hand_constraint_weight = 1.0F;
    const ControllerInteractionHandConstraint switched_constraint =
        make_hand_constraint(switched, entry, Hand::Left);
    const ControllerInteractionFrameState switched_frame = handoff.apply(
        entry,
        switched,
        interaction::kControllerStepSeconds,
        switched_constraint);
    require(
        !switched_frame.hand_constraint_validated &&
            !switched_frame.hand_constraint_result.applied,
        "switched identity started the first constraint inside ownership");

    owned.diagnostics.state = RuntimeState::Carry;
    owned.diagnostics.object_state = ObjectState::Held;
    owned.diagnostics.attached = true;
    owned.diagnostics.recorded_carry = true;
    owned.diagnostics.hand_constraint_weight = 1.0F;
    const ControllerInteractionHandConstraint stale_original =
        make_hand_constraint(owned, entry, Hand::Right);
    const ControllerInteractionFrameState rolled_back = handoff.apply(
        entry,
        owned,
        interaction::kControllerStepSeconds,
        stale_original);
    require(
        !rolled_back.hand_constraint_validated &&
            !rolled_back.hand_constraint_result.applied,
        "poisoned ownership identity resurrected after rollback");
}

void test_delayed_constraint_rejects_free_lifecycle() {
    {
        const FlatControllerPose entry = make_flat_pose();
        const Pose raw_reference = interaction::expand_flat_controller_pose(
            entry, make_pose(0.375F));
        RuntimeOutput owned = make_owned_output(raw_reference);
        owned.diagnostics.target = {723U, 7U};
        owned.diagnostics.affordance_id = 103U;
        owned.diagnostics.hand = Hand::Right;
        owned.diagnostics.object_state = ObjectState::Targeted;

        ControllerInteractionFrameHandoff handoff;
        (void)handoff.apply(
            entry,
            owned,
            interaction::kControllerStepSeconds,
            std::nullopt);

        owned.diagnostics.state = RuntimeState::PickupReplay;
        owned.diagnostics.object_state = ObjectState::Free;
        owned.diagnostics.attached = false;
        owned.diagnostics.hand_constraint_weight = 1.0F;
        const ControllerInteractionHandConstraint constraint =
            make_hand_constraint(owned, entry, Hand::Right);
        const ControllerInteractionFrameState invalid = handoff.apply(
            entry,
            owned,
            interaction::kControllerStepSeconds,
            constraint);
        require(
            !invalid.hand_constraint_validated &&
                !invalid.hand_constraint_result.applied,
            "Free delayed constraint started before attachment");

        owned.diagnostics.object_state = ObjectState::Attached;
        owned.diagnostics.attached = true;
        const ControllerInteractionFrameState attached = handoff.apply(
            entry,
            owned,
            interaction::kControllerStepSeconds,
            constraint);
        require(
            attached.hand_constraint_validated &&
                attached.hand_constraint_result.applied &&
                attached.hand_constraint_result.reachable,
            "Attached state did not start after Free lifecycle rejection");
    }
}

void test_delayed_constraint_accepts_targeted_lifecycle() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput owned = make_owned_output(raw_reference);
    owned.diagnostics.target = {724U, 8U};
    owned.diagnostics.affordance_id = 104U;
    owned.diagnostics.hand = Hand::Right;
    owned.diagnostics.object_state = ObjectState::Targeted;
    owned.diagnostics.attached = false;

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        entry,
        owned,
        interaction::kControllerStepSeconds,
        std::nullopt);

    owned.diagnostics.hand_constraint_weight = 0.04F;
    const ControllerInteractionHandConstraint constraint =
        make_hand_constraint(owned, entry, Hand::Right);
    const ControllerInteractionFrameState targeted = handoff.apply(
        entry,
        owned,
        interaction::kControllerStepSeconds,
        constraint);
    require(
        targeted.hand_constraint_validated &&
            targeted.hand_constraint_result.applied &&
            targeted.hand_constraint_result.reachable,
        "delayed positive-weight Targeted constraint did not start inside "
        "the owned reach epoch");
}

void test_hand_constraint_keeps_layered_carry_lower_body_exact_and_selected_only() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput owned = make_owned_output(raw_reference);
    owned.diagnostics.target = {72U, 5U};
    owned.diagnostics.affordance_id = 10U;
    owned.diagnostics.hand = Hand::Left;
    owned.diagnostics.state = RuntimeState::Hold;
    owned.diagnostics.object_state = ObjectState::Held;
    owned.diagnostics.attached = true;
    owned.diagnostics.hand_constraint_weight = 0.0F;
    const ControllerInteractionHandConstraint constraint =
        make_hand_constraint(owned, entry, Hand::Left);

    ControllerInteractionFrameHandoff baseline_handoff;
    ControllerInteractionFrameHandoff constrained_handoff;
    (void)baseline_handoff.apply(
        entry, owned, interaction::kControllerStepSeconds, std::nullopt);
    (void)constrained_handoff.apply(
        entry, owned, interaction::kControllerStepSeconds, constraint);

    FlatControllerPose fresh_locomotion = entry;
    for (size_t bone = 0; bone <= 9U; ++bone) {
        const float value = static_cast<float>(bone + 1U);
        fresh_locomotion.positions[bone] =
            fresh_locomotion.positions[bone] +
            vec3(0.02F * value, -0.01F * value, 0.03F * value);
        fresh_locomotion.velocities[bone] =
            fresh_locomotion.velocities[bone] +
            vec3(-0.04F * value, 0.01F * value, 0.02F * value);
        fresh_locomotion.rotations[bone] = quat_from_angle_axis(
            0.01F * value, vec3(0.0F, 1.0F, 0.0F));
        fresh_locomotion.angular_velocities[bone] =
            fresh_locomotion.angular_velocities[bone] +
            vec3(0.03F * value, -0.02F * value, 0.01F * value);
    }
    for (size_t bone = 19U; bone <= 22U; ++bone) {
        const float value = static_cast<float>(bone + 1U);
        fresh_locomotion.positions[bone] =
            fresh_locomotion.positions[bone] +
            vec3(0.004F * value, -0.003F * value, 0.002F * value);
        fresh_locomotion.velocities[bone] =
            vec3(-0.08F * value, 0.06F * value, -0.05F * value);
        fresh_locomotion.rotations[bone] = quat_from_angle_axis(
            0.013F * value,
            normalize(vec3(0.7F, 0.2F, 0.4F + 0.01F * value)));
        fresh_locomotion.angular_velocities[bone] =
            vec3(0.05F * value, -0.03F * value, 0.07F * value);
    }
    fresh_locomotion.foot_contacts = {0U, 1U};
    owned.diagnostics.state = RuntimeState::Carry;
    owned.diagnostics.recorded_carry = false;
    owned.diagnostics.inactive_arm_targets_locomotion = true;
    owned.diagnostics.inactive_arm_tracks_locomotion = true;
    owned.diagnostics.hand_constraint_weight = 0.0F;
    const ControllerInteractionHandConstraint carry_constraint =
        make_hand_constraint(owned, fresh_locomotion, Hand::Left);

    ControllerInteractionFrameState baseline{};
    ControllerInteractionFrameState constrained{};
    for (int tick = 0; tick < 32; ++tick) {
        baseline = baseline_handoff.apply(
            fresh_locomotion,
            owned,
            interaction::kControllerStepSeconds,
            std::nullopt);
        constrained = constrained_handoff.apply(
            fresh_locomotion,
            owned,
            interaction::kControllerStepSeconds,
            carry_constraint);
    }
    owned.diagnostics.hand_constraint_weight = 1.0F;
    baseline = baseline_handoff.apply(
        fresh_locomotion,
        owned,
        interaction::kControllerStepSeconds,
        std::nullopt);
    constrained = constrained_handoff.apply(
        fresh_locomotion,
        owned,
        interaction::kControllerStepSeconds,
        carry_constraint);
    require(
        constrained.hand_constraint_result.applied &&
            constrained.hand_constraint_result.reachable,
        "layered Carry did not apply the selected hand constraint");
    bool selected_chain_changed = false;
    for (size_t bone = 0; bone < constrained.pose.positions.size(); ++bone) {
        const bool selected_chain = bone >= 15U && bone <= 18U;
        require(
            vec_bits_equal(
                constrained.pose.positions[bone], baseline.pose.positions[bone]) &&
                vec_bits_equal(
                    constrained.pose.velocities[bone], baseline.pose.velocities[bone]),
            "layered hand constraint changed a translation channel");
        if (selected_chain) {
            selected_chain_changed = selected_chain_changed ||
                !quat_bits_equal(
                    constrained.pose.rotations[bone],
                    baseline.pose.rotations[bone]) ||
                !vec_bits_equal(
                    constrained.pose.angular_velocities[bone],
                    baseline.pose.angular_velocities[bone]);
            continue;
        }
        require(
            quat_bits_equal(
                constrained.pose.rotations[bone], baseline.pose.rotations[bone]) &&
                vec_bits_equal(
                    constrained.pose.angular_velocities[bone],
                    baseline.pose.angular_velocities[bone]),
            "layered hand constraint changed a non-selected rotation channel");
    }
    require(selected_chain_changed, "selected layered arm chain did not change");
    for (size_t bone = 0; bone <= 9U; ++bone) {
        require(
            vec_bits_equal(
                constrained.pose.positions[bone],
                fresh_locomotion.positions[bone]) &&
                vec_bits_equal(
                    constrained.pose.velocities[bone],
                    fresh_locomotion.velocities[bone]) &&
                quat_bits_equal(
                    constrained.pose.rotations[bone],
                    fresh_locomotion.rotations[bone]) &&
                vec_bits_equal(
                    constrained.pose.angular_velocities[bone],
                    fresh_locomotion.angular_velocities[bone]),
            "layered hand constraint changed live locomotion bones 0..9");
    }
    require(
        constrained.pose.foot_contacts == fresh_locomotion.foot_contacts,
        "layered hand constraint changed live locomotion contacts");
    for (size_t bone = 19U; bone <= 22U; ++bone) {
        require(
            flat_bone_channels_bits_equal(
                constrained.pose, fresh_locomotion, bone),
            "selected-hand IK changed released inactive-arm locomotion");
    }
}

void test_layered_carry_final_composite_guard_is_atomic_for_both_hands() {
    for (const Hand hand : {Hand::Left, Hand::Right}) {
        const FlatControllerPose entry = make_flat_pose();
        const Pose raw_reference = interaction::expand_flat_controller_pose(
            entry, make_pose(0.375F));
        RuntimeOutput output = make_owned_output(raw_reference);
        output.diagnostics.state = RuntimeState::Hold;
        output.diagnostics.target = {
            hand == Hand::Left ? 801U : 802U,
            hand == Hand::Left ? 11U : 12U};
        output.diagnostics.affordance_id =
            hand == Hand::Left ? 31U : 32U;
        output.diagnostics.hand = hand;
        output.diagnostics.object_state = ObjectState::Held;
        output.diagnostics.attached = true;
        output.diagnostics.hand_constraint_weight = 0.0F;
        ControllerInteractionHandConstraint constraint =
            make_hand_constraint(output, entry, hand);
        const FlatWorldPose entry_world = flat_world_pose(entry);
        const size_t selected_upper_arm =
            hand == Hand::Left ? 16U : 20U;
        const size_t selected_hand =
            hand == Hand::Left ? 18U : 22U;
        constraint.grasp_world = {
            lerp(
                entry_world.positions[selected_upper_arm],
                entry_world.positions[selected_hand],
                0.98F),
            entry_world.rotations[selected_hand]};

        ControllerInteractionFrameHandoff actual;
        ControllerInteractionFrameHandoff oracle;
        const ControllerInteractionFrameState actual_entry = actual.apply(
            entry,
            output,
            interaction::kControllerStepSeconds,
            constraint);
        const ControllerInteractionFrameState oracle_entry = oracle.apply(
            entry,
            output,
            interaction::kControllerStepSeconds,
            constraint);
        require(
            frame_state_bits_equal(actual_entry, oracle_entry),
            "final-composite guard setup diverged before layered Carry");

        FlatControllerPose locomotion = entry;
        locomotion.positions[0] =
            locomotion.positions[0] + vec3(0.12F, 0.0F, -0.04F);
        locomotion.rotations[1] = quat_from_angle_axis(
            0.20F, normalize(vec3(0.2F, 0.9F, 0.3F)));
        const size_t inactive_arm_begin =
            hand == Hand::Left ? 19U : 15U;
        for (size_t bone = inactive_arm_begin;
             bone < inactive_arm_begin + 4U;
             ++bone) {
            locomotion.rotations[bone] = quat_from_angle_axis(
                0.16F + 0.01F * static_cast<float>(bone - inactive_arm_begin),
                normalize(vec3(0.3F, 0.7F, 0.2F)));
        }
        locomotion.foot_contacts = {0U, 1U};

        FlatControllerPose safe_upper = entry;
        safe_upper.rotations[12] = quat_from_angle_axis(
            0.12F, normalize(vec3(0.2F, 0.8F, 0.5F)));
        output.pose = interaction::expand_flat_controller_pose(
            safe_upper, raw_reference);
        output.diagnostics.state = RuntimeState::Carry;
        output.diagnostics.recorded_carry = false;
        output.diagnostics.inactive_arm_targets_locomotion = true;
        output.diagnostics.inactive_arm_tracks_locomotion = true;
        output.diagnostics.hand_constraint_weight = 1.0F;

        const ControllerInteractionFrameState first_actual = actual.apply(
            locomotion,
            output,
            interaction::kControllerStepSeconds,
            constraint);
        const ControllerInteractionFrameState first_oracle = oracle.apply(
            locomotion,
            output,
            interaction::kControllerStepSeconds,
            constraint);
        require(
            frame_state_bits_equal(first_actual, first_oracle),
            "normal lower/upper/IK composite diverged between identical handoffs");
        require(
            first_actual.hand_constraint_result.applied &&
                first_actual.hand_constraint_result.reachable,
            "normal lower/upper/IK composite did not solve the selected hand");

        const auto require_bounded_final_fk = [](
            const FlatControllerPose& previous,
            const FlatControllerPose& current) {
            const FlatWorldPose previous_world = flat_world_pose(previous);
            const FlatWorldPose current_world = flat_world_pose(current);
            for (size_t bone = 0;
                 bone < interaction::kFlatControllerBoneCount;
                 ++bone) {
                require(
                    length(
                        current_world.positions[bone] -
                        previous_world.positions[bone]) <= 0.20F,
                    "normal final composite exceeded the FK translation bound");
                require(
                    rotation_distance(
                        current_world.rotations[bone],
                        previous_world.rotations[bone]) <=
                        60.0F * 3.14159265358979323846F / 180.0F,
                    "normal final composite exceeded the FK rotation bound");
            }
        };
        require_bounded_final_fk(entry, first_actual.pose);

        RuntimeOutput adversarial = output;
        FlatControllerPose adversarial_upper = safe_upper;
        adversarial_upper.rotations[12] = quat_from_angle_axis(
            2.40F, normalize(vec3(0.2F, 0.8F, 0.5F)));
        adversarial.pose = interaction::expand_flat_controller_pose(
            adversarial_upper, raw_reference);
        const RuntimeOutput adversarial_before = adversarial;
        bool threw = false;
        try {
            (void)actual.apply(
                locomotion,
                adversarial,
                interaction::kControllerStepSeconds,
                constraint);
        } catch (const interaction::FormatError&) {
            threw = true;
        }
        require(
            threw,
            "final composite guard accepted an adversarial upper-body step");
        require(
            output_fields_equal(adversarial, adversarial_before),
            "rejected final composite mutated its runtime input");

        const ControllerInteractionFrameState expected_retry = oracle.apply(
            locomotion,
            output,
            interaction::kControllerStepSeconds,
            constraint);
        const ControllerInteractionFrameState actual_retry = actual.apply(
            locomotion,
            output,
            interaction::kControllerStepSeconds,
            constraint);
        require(
            frame_state_bits_equal(actual_retry, expected_retry),
            "rejected final composite did not restore handoff state atomically");
        require_bounded_final_fk(first_actual.pose, actual_retry.pose);
    }
}

void test_hand_constraint_mismatch_or_invalid_input_disables_atomically() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput owned = make_owned_output(raw_reference);
    owned.diagnostics.target = {73U, 6U};
    owned.diagnostics.affordance_id = 11U;
    owned.diagnostics.hand = Hand::Right;
    owned.diagnostics.state = RuntimeState::Hold;
    owned.diagnostics.object_state = ObjectState::Held;
    owned.diagnostics.attached = true;
    owned.diagnostics.hand_constraint_weight = 0.0F;
    const ControllerInteractionHandConstraint valid =
        make_hand_constraint(owned, entry, Hand::Right);

    std::array<std::optional<ControllerInteractionHandConstraint>, 6>
        invalid_constraints{};
    invalid_constraints[0] = std::nullopt;
    invalid_constraints[1] = valid;
    invalid_constraints[1]->target.generation += 1U;
    invalid_constraints[2] = valid;
    invalid_constraints[2]->affordance_id += 1U;
    invalid_constraints[3] = valid;
    invalid_constraints[3]->hand = Hand::Left;
    invalid_constraints[4] = valid;
    invalid_constraints[4]->grasp_world.position.x =
        std::numeric_limits<float>::quiet_NaN();
    invalid_constraints[5] = valid;
    invalid_constraints[5]->grasp_world.rotation =
        quat(0.0F, 0.0F, 0.0F, 0.0F);

    for (const auto& invalid : invalid_constraints) {
        ControllerInteractionFrameHandoff handoff;
        (void)handoff.apply(
            entry, owned, interaction::kControllerStepSeconds, valid);
        RuntimeOutput full_weight = owned;
        full_weight.diagnostics.hand_constraint_weight = 1.0F;
        const RuntimeOutput input_before = full_weight;
        const ControllerInteractionFrameState frame = handoff.apply(
            entry,
            full_weight,
            interaction::kControllerStepSeconds,
            invalid);
        require(
            !frame.hand_constraint_result.applied &&
                !frame.hand_constraint_validated &&
                flat_pose_bits_equal(frame.pose, entry) &&
                output_fields_equal(full_weight, input_before),
            "mismatched or invalid constraint partially changed the frame");
    }

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        entry, owned, interaction::kControllerStepSeconds, valid);
    RuntimeOutput switched = owned;
    switched.diagnostics.target = {74U, 1U};
    switched.diagnostics.affordance_id = 12U;
    switched.diagnostics.hand = Hand::Left;
    switched.diagnostics.hand_constraint_weight = 1.0F;
    ControllerInteractionHandConstraint matching_switched =
        make_hand_constraint(switched, entry, Hand::Left);
    const ControllerInteractionFrameState switched_frame = handoff.apply(
        entry,
        switched,
        interaction::kControllerStepSeconds,
        matching_switched);
    require(
        !switched_frame.hand_constraint_result.applied &&
            !switched_frame.hand_constraint_validated &&
            flat_pose_bits_equal(switched_frame.pose, entry),
        "selection changed inside an ownership epoch");
}

void test_hand_constraint_reset_and_reentry_recalibrates_selected_hand() {
    const FlatControllerPose first_entry = make_flat_pose();
    const Pose first_raw_reference = interaction::expand_flat_controller_pose(
        first_entry, make_pose(0.375F));
    RuntimeOutput first_owned = make_owned_output(first_raw_reference);
    first_owned.diagnostics.target = {75U, 1U};
    first_owned.diagnostics.affordance_id = 13U;
    first_owned.diagnostics.hand = Hand::Right;
    first_owned.diagnostics.state = RuntimeState::Hold;
    first_owned.diagnostics.object_state = ObjectState::Held;
    first_owned.diagnostics.attached = true;
    first_owned.diagnostics.hand_constraint_weight = 0.0F;
    const ControllerInteractionHandConstraint first_constraint =
        make_hand_constraint(first_owned, first_entry, Hand::Right);
    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        first_entry,
        first_owned,
        interaction::kControllerStepSeconds,
        first_constraint);
    first_owned.diagnostics.hand_constraint_weight = 1.0F;
    require(
        handoff.apply(
            first_entry,
            first_owned,
            interaction::kControllerStepSeconds,
            first_constraint).hand_constraint_result.applied,
        "first epoch did not seed the right-hand solver");

    RuntimeOutput idle;
    bool released = false;
    for (int frame = 0; frame < 20; ++frame) {
        const ControllerInteractionFrameState release = handoff.apply(
            first_entry,
            idle,
            interaction::kControllerStepSeconds,
            std::nullopt);
        if (!release.overrides_locomotion_pose) {
            released = true;
            break;
        }
    }
    require(released, "first epoch release did not complete");

    FlatControllerPose second_entry = first_entry;
    second_entry.rotations[12] = quat_from_angle_axis(
        0.45F, normalize(vec3(0.2F, 0.8F, 0.4F)));
    const Pose second_raw_reference = interaction::expand_flat_controller_pose(
        second_entry, make_pose(2.5F));
    RuntimeOutput second_owned = make_owned_output(second_raw_reference);
    second_owned.diagnostics.target = {76U, 2U};
    second_owned.diagnostics.affordance_id = 14U;
    second_owned.diagnostics.hand = Hand::Left;
    second_owned.diagnostics.state = RuntimeState::Hold;
    second_owned.diagnostics.object_state = ObjectState::Held;
    second_owned.diagnostics.attached = true;
    second_owned.diagnostics.hand_constraint_weight = 0.0F;
    const ControllerInteractionHandConstraint second_constraint =
        make_hand_constraint(second_owned, second_entry, Hand::Left);
    const ControllerInteractionFrameState second_first = handoff.apply(
        second_entry,
        second_owned,
        interaction::kControllerStepSeconds,
        second_constraint);
    require(
        flat_pose_bits_equal(second_first.pose, second_entry),
        "re-entry did not capture a fresh flat calibration reference");
    second_owned.diagnostics.hand_constraint_weight = 1.0F;
    const ControllerInteractionFrameState second_solved = handoff.apply(
        second_entry,
        second_owned,
        interaction::kControllerStepSeconds,
        second_constraint);
    require(
        second_solved.hand_constraint_result.applied &&
            second_solved.hand_constraint_result.reachable,
        "re-entry did not calibrate the newly selected left hand");
    const FlatWorldPose second_world = flat_world_pose(second_solved.pose);
    require_vec_near(
        second_world.positions[18],
        second_constraint.grasp_world.position,
        "re-entered left hand missed the semantic grasp",
        3.0e-4F);
    for (size_t bone = 19U; bone <= 22U; ++bone) {
        require(
            quat_bits_equal(
                second_solved.pose.rotations[bone],
                second_entry.rotations[bone]) &&
                vec_bits_equal(
                    second_solved.pose.angular_velocities[bone],
                    second_entry.angular_velocities[bone]),
            "re-entry reused the prior right-hand selection");
    }
}

void test_frame_handoff_crosses_179_9_to_180_1_incrementally() {
    FlatControllerPose entry = make_flat_pose();
    entry.rotations[0] = quat();
    RuntimeOutput owned = make_owned_output(make_pose(0.0F));
    owned.pose.rotations[0] = quat();
    ControllerInteractionFrameHandoff handoff;
    ControllerInteractionFrameState previous = handoff.apply(
        entry, owned, interaction::kControllerStepSeconds);
    for (float degrees :
         {30.0F, 60.0F, 90.0F, 120.0F, 150.0F, 179.9F, 180.1F}) {
        owned.pose.rotations[0] = quat_from_angle_axis(
            degrees * 3.14159265358979323846F / 180.0F,
            vec3(0.0F, 1.0F, 0.0F));
        const ControllerInteractionFrameState frame = handoff.apply(
            entry, owned, interaction::kControllerStepSeconds);
        require(
            rotation_distance(
                previous.pose.rotations[0], frame.pose.rotations[0]) <
                31.0F * 3.14159265358979323846F / 180.0F,
            "moving target switched the old fixed-source branch");
        previous = frame;
    }
}

void test_frame_handoff_keeps_captured_neck_and_head_while_owned() {
    FlatControllerPose entry = make_flat_pose();
    RuntimeOutput owned = make_owned_output(make_pose(0.375F));

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(entry, owned, interaction::kControllerStepSeconds);
    owned.pose.rotations[16] = quat_mul(
        quat_from_angle_axis(
            0.65F, normalize(vec3(0.2F, 0.8F, 0.3F))),
        owned.pose.rotations[16]);

    FlatControllerPose fresh_locomotion = entry;
    for (size_t bone : {13U, 14U}) {
        fresh_locomotion.positions[bone] = fresh_locomotion.positions[bone] +
            vec3(50.0F, -30.0F, 70.0F);
        fresh_locomotion.velocities[bone] = fresh_locomotion.velocities[bone] +
            vec3(-20.0F, 40.0F, 80.0F);
        fresh_locomotion.rotations[bone] = quat_from_angle_axis(
            2.4F, normalize(vec3(1.0F, 2.0F, 3.0F)));
        fresh_locomotion.angular_velocities[bone] =
            fresh_locomotion.angular_velocities[bone] +
            vec3(90.0F, 60.0F, -40.0F);
    }
    const ControllerInteractionFrameState frame = handoff.apply(
        fresh_locomotion, owned, interaction::kControllerStepSeconds);

    for (size_t bone : {13U, 14U}) {
        require_vec_near(
            frame.pose.positions[bone],
            entry.positions[bone],
            "fresh locomotion moved an owned Neck/Head position");
        require_vec_near(
            frame.pose.velocities[bone],
            entry.velocities[bone],
            "fresh locomotion moved an owned Neck/Head velocity");
        require_same_rotation(
            frame.pose.rotations[bone],
            entry.rotations[bone],
            "fresh locomotion moved an owned Neck/Head rotation");
        require_vec_near(
            frame.pose.angular_velocities[bone],
            entry.angular_velocities[bone],
            "fresh locomotion moved an owned Neck/Head angular velocity");
    }
}

float rotation_distance(quat left, quat right) {
    const float orientation_dot = std::clamp(
        std::fabs(quat_dot(quat_normalize(left), quat_normalize(right))),
        0.0F,
        1.0F);
    return 2.0F * std::acos(orientation_dot);
}

void test_hold_to_layered_carry_first_frame_preserves_rendered_lower_body() {
    FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::Hold;

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        entry, output, interaction::kControllerStepSeconds);

    FlatControllerPose held = entry;
    held.positions[0] = held.positions[0] + vec3(0.20F, 0.0F, -0.15F);
    held.rotations[1] = quat_from_angle_axis(
        0.35F, normalize(vec3(0.2F, 0.9F, 0.3F)));
    held.rotations[3] = quat_from_angle_axis(
        -0.45F, normalize(vec3(0.7F, 0.2F, 0.4F)));
    held.rotations[7] = quat_from_angle_axis(
        0.30F, normalize(vec3(0.3F, 0.5F, 0.8F)));
    output.pose = interaction::expand_flat_controller_pose(
        held, raw_reference);
    const ControllerInteractionFrameState before_carry = handoff.apply(
        entry, output, interaction::kControllerStepSeconds);
    require_flat_pose_near(
        before_carry.pose,
        held,
        "Hold setup did not publish the authored full-body pose");

    FlatControllerPose locomotion = entry;
    locomotion.positions[0] =
        locomotion.positions[0] + vec3(-0.30F, 0.0F, 0.25F);
    locomotion.rotations[1] = quat_from_angle_axis(
        -0.40F, normalize(vec3(0.4F, 0.8F, 0.1F)));
    locomotion.rotations[3] = quat_from_angle_axis(
        0.55F, normalize(vec3(0.6F, 0.3F, 0.7F)));
    locomotion.rotations[7] = quat_from_angle_axis(
        -0.35F, normalize(vec3(0.2F, 0.7F, 0.6F)));
    locomotion.foot_contacts = {0U, 1U};
    const FlatControllerPose locomotion_before = locomotion;
    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = false;
    const RuntimeOutput output_before = output;
    const ControllerInteractionFrameState first_carry = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);

    for (size_t bone = 0; bone <= 9U; ++bone) {
        require(
            flat_bone_channels_bits_equal(
                first_carry.pose, before_carry.pose, bone),
            "first layered Carry frame snapped a rendered lower-body channel");
    }
    require(
        first_carry.pose.foot_contacts == locomotion.foot_contacts,
        "first layered Carry frame did not publish live locomotion contacts");
    require(
        !first_carry.synchronize_simulation_root,
        "layered lower-body handoff overwrote the live simulation root");
    require(
        flat_pose_bits_equal(locomotion, locomotion_before),
        "layered lower-body handoff mutated the locomotion input");
    require(
        output_fields_equal(output, output_before),
        "layered lower-body handoff mutated the runtime input");
}

void test_layered_carry_lower_body_rebases_moving_target_and_cleans_up() {
    FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::Hold;

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        entry, output, interaction::kControllerStepSeconds);

    FlatControllerPose held = entry;
    held.positions[0] = held.positions[0] + vec3(0.15F, 0.0F, -0.10F);
    held.rotations[1] = quat_from_angle_axis(
        0.30F, normalize(vec3(0.1F, 0.9F, 0.2F)));
    held.rotations[3] = quat_from_angle_axis(
        -0.35F, normalize(vec3(0.8F, 0.2F, 0.3F)));
    output.pose = interaction::expand_flat_controller_pose(
        held, raw_reference);
    ControllerInteractionFrameState previous = handoff.apply(
        entry, output, interaction::kControllerStepSeconds);

    FlatControllerPose locomotion = entry;
    locomotion.positions[0] =
        locomotion.positions[0] + vec3(-0.25F, 0.0F, 0.20F);
    locomotion.rotations[1] = quat_from_angle_axis(
        -0.25F, normalize(vec3(0.2F, 0.8F, 0.3F)));
    locomotion.rotations[3] = quat_from_angle_axis(
        0.40F, normalize(vec3(0.7F, 0.3F, 0.5F)));
    locomotion.foot_contacts = {0U, 1U};
    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = false;

    ControllerInteractionFrameState frame = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    for (size_t bone = 0; bone <= 9U; ++bone) {
        require(
            flat_bone_channels_bits_equal(
                frame.pose, previous.pose, bone),
            "moving-target handoff snapped on its zero-decay frame");
    }

    const auto require_bounded_final_fk = [&] {
        const FlatWorldPose previous_world = flat_world_pose(previous.pose);
        const FlatWorldPose current_world = flat_world_pose(frame.pose);
        for (size_t bone = 0;
             bone < interaction::kFlatControllerBoneCount;
             ++bone) {
            require(
                length(
                    current_world.positions[bone] -
                    previous_world.positions[bone]) <= 0.20F,
                "layered lower-body handoff exceeded the final-FK translation bound");
            require(
                rotation_distance(
                    current_world.rotations[bone],
                    previous_world.rotations[bone]) <=
                    60.0F * 3.14159265358979323846F / 180.0F,
                "layered lower-body handoff exceeded the final-FK rotation bound");
        }
    };
    require_bounded_final_fk();
    previous = frame;

    for (int tick = 1; tick <= 4; ++tick) {
        locomotion.positions[0].x += 0.008F;
        locomotion.positions[0].z -= 0.004F;
        locomotion.rotations[7] = quat_from_angle_axis(
            0.015F * static_cast<float>(tick),
            normalize(vec3(0.2F, 0.7F, 0.5F)));
        locomotion.foot_contacts = {
            static_cast<uint8_t>(tick % 2),
            static_cast<uint8_t>((tick + 1) % 2)};
        const FlatControllerPose locomotion_before = locomotion;
        const RuntimeOutput output_before = output;
        frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        require_bounded_final_fk();
        require(
            frame.pose.foot_contacts == locomotion.foot_contacts,
            "moving lower-body handoff cached stale contacts");
        require(
            flat_pose_bits_equal(locomotion, locomotion_before) &&
                output_fields_equal(output, output_before),
            "moving lower-body handoff mutated an input");
        previous = frame;
    }

    locomotion.positions[0].x += 1.40F;
    locomotion.rotations[1] = quat_from_angle_axis(
        1.20F, normalize(vec3(0.1F, 0.9F, 0.3F)));
    locomotion.foot_contacts = {1U, 0U};
    frame = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    require_bounded_final_fk();
    require(
        !flat_bone_channels_bits_equal(frame.pose, locomotion, 0U),
        "large locomotion replan bypassed lower-body inertialization");
    require(
        frame.pose.foot_contacts == locomotion.foot_contacts &&
            !frame.synchronize_simulation_root,
        "lower-body rebase stole contacts or simulation-root authority");
    previous = frame;

    for (int tick = 0; tick < 4; ++tick) {
        locomotion.positions[0].z += 0.002F;
        locomotion.foot_contacts = {
            static_cast<uint8_t>((tick + 1) % 2),
            static_cast<uint8_t>(tick % 2)};
        frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        require_bounded_final_fk();
        require(
            frame.pose.foot_contacts == locomotion.foot_contacts,
            "rebased lower-body handoff cached stale contacts");
        previous = frame;
    }

    bool converged = false;
    for (int tick = 0; tick < 64 && !converged; ++tick) {
        frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        require_bounded_final_fk();
        converged = true;
        for (size_t bone = 0; bone <= 9U; ++bone) {
            converged = converged && flat_bone_channels_bits_equal(
                frame.pose, locomotion, bone);
        }
        previous = frame;
    }
    require(
        converged,
        "rebased lower-body handoff did not converge after target stabilization");

    locomotion.positions[0].x += 0.006F;
    locomotion.rotations[7] = quat_from_angle_axis(
        0.09F, normalize(vec3(0.2F, 0.7F, 0.5F)));
    locomotion.foot_contacts = {0U, 1U};
    frame = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    for (size_t bone = 0; bone <= 9U; ++bone) {
        require(
            flat_bone_channels_bits_equal(frame.pose, locomotion, bone),
            "completed lower-body handoff retained stale transition state");
    }
    require(
        frame.pose.foot_contacts == locomotion.foot_contacts,
        "completed lower-body handoff retained stale contacts");
}

void test_layered_carry_deadline_target_change_rebases_without_exact_snap() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::Hold;

    ControllerInteractionFrameHandoff handoff;
    ControllerInteractionFrameState previous = handoff.apply(
        entry, output, interaction::kControllerStepSeconds);

    FlatControllerPose displaced_hold = entry;
    displaced_hold.positions[0].x += 0.20F;
    output.pose = interaction::expand_flat_controller_pose(
        displaced_hold, raw_reference);
    previous = handoff.apply(
        entry, output, interaction::kControllerStepSeconds);
    require(
        float_bits_equal(
            previous.pose.positions[0].x,
            displaced_hold.positions[0].x),
        "deadline rebase setup did not publish the displaced Hold root");

    FlatControllerPose locomotion = entry;
    locomotion.foot_contacts = {0U, 1U};
    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = false;
    previous = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    for (int tick = 0; tick < 12; ++tick) {
        previous = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
    }

    locomotion.positions[0].x += 0.199F;
    locomotion.foot_contacts = {1U, 0U};
    const ControllerInteractionFrameState deadline = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    require(
        !float_bits_equal(
            deadline.pose.positions[0].x,
            locomotion.positions[0].x),
        "0.50 second deadline forced an exact root scalar snap");
    require(
        !flat_bone_channels_bits_equal(deadline.pose, locomotion, 0U),
        "0.50 second deadline forced an exact root pose snap");
    for (size_t bone = 0; bone <= 9U; ++bone) {
        require(
            flat_bone_channels_bits_equal(deadline.pose, previous.pose, bone),
            "deadline target replan took a broad lower-body trial step "
            "instead of rebasing");
    }
    const FlatWorldPose previous_world = flat_world_pose(previous.pose);
    const FlatWorldPose deadline_world = flat_world_pose(deadline.pose);
    for (size_t bone = 0;
         bone < interaction::kFlatControllerBoneCount;
         ++bone) {
        require(
            length(
                deadline_world.positions[bone] -
                previous_world.positions[bone]) <= 0.20F,
            "deadline target rebase exceeded the final-FK translation bound");
        require(
            rotation_distance(
                deadline_world.rotations[bone],
                previous_world.rotations[bone]) <=
                60.0F * 3.14159265358979323846F / 180.0F,
            "deadline target rebase exceeded the final-FK rotation bound");
    }
    require(
        deadline.pose.foot_contacts == locomotion.foot_contacts &&
            !deadline.synchronize_simulation_root,
        "deadline target rebase stole contacts or simulation-root authority");

    bool converged = false;
    ControllerInteractionFrameState frame = deadline;
    for (int tick = 0; tick < 64 && !converged; ++tick) {
        frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        converged = true;
        for (size_t bone = 0; bone <= 9U; ++bone) {
            converged = converged && flat_bone_channels_bits_equal(
                frame.pose, locomotion, bone);
        }
    }
    require(
        converged,
        "deadline-rebased lower body did not eventually converge bit-exactly");
}

void test_layered_carry_terminal_requires_velocity_offsets_to_decay() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::Hold;

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        entry, output, interaction::kControllerStepSeconds);

    FlatControllerPose moving_hold = entry;
    moving_hold.velocities[0].x = 1.45F;
    moving_hold.angular_velocities[1].y = 8.0F;
    output.pose = interaction::expand_flat_controller_pose(
        moving_hold, raw_reference);
    const ControllerInteractionFrameState held = handoff.apply(
        entry, output, interaction::kControllerStepSeconds);
    require(
        length(held.pose.velocities[0]) > 1.0F &&
            length(held.pose.angular_velocities[1]) > 5.0F,
        "velocity-offset setup did not reach the rendered Hold pose");

    FlatControllerPose locomotion = entry;
    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = false;
    (void)handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    for (int tick = 0; tick < 12; ++tick) {
        (void)handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
    }

    const ControllerInteractionFrameState deadline = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    require(
        !vec_bits_equal(
            deadline.pose.velocities[0], locomotion.velocities[0]),
        "nominal deadline discarded a residual linear-velocity offset");
    require(
        !vec_bits_equal(
            deadline.pose.angular_velocities[1],
            locomotion.angular_velocities[1]),
        "nominal deadline discarded a residual angular-velocity offset");

    bool converged = false;
    ControllerInteractionFrameState frame = deadline;
    for (int tick = 0; tick < 64 && !converged; ++tick) {
        frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        converged = true;
        for (size_t bone = 0; bone <= 9U; ++bone) {
            converged = converged && flat_bone_channels_bits_equal(
                frame.pose, locomotion, bone);
        }
    }
    require(
        converged,
        "velocity-offset handoff did not eventually converge bit-exactly");
}

void test_layered_carry_terminal_requires_position_and_rotation_offsets_to_decay() {
    const auto run_residual_case = [](bool rotational_residual) {
        const FlatControllerPose entry = make_flat_pose();
        const Pose raw_reference = interaction::expand_flat_controller_pose(
            entry, make_pose(0.375F));
        RuntimeOutput output = make_owned_output(raw_reference);
        output.diagnostics.state = RuntimeState::Hold;

        ControllerInteractionFrameHandoff handoff;
        (void)handoff.apply(
            entry, output, interaction::kControllerStepSeconds);

        FlatControllerPose held = entry;
        const size_t residual_bone = rotational_residual ? 1U : 0U;
        if (rotational_residual) {
            held.rotations[residual_bone] = quat_normalize(quat_mul(
                quat_from_angle_axis(
                    1.0F * 3.14159265358979323846F / 180.0F,
                    vec3(0.0F, 1.0F, 0.0F)),
                held.rotations[residual_bone]));
        } else {
            held.positions[residual_bone].x += 0.002F;
        }
        output.pose = interaction::expand_flat_controller_pose(
            held, raw_reference);
        const ControllerInteractionFrameState held_frame = handoff.apply(
            entry, output, interaction::kControllerStepSeconds);

        FlatControllerPose locomotion = entry;
        output.diagnostics.state = RuntimeState::Carry;
        output.diagnostics.recorded_carry = false;
        const ControllerInteractionFrameState first_carry = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        require(
            flat_bone_channels_bits_equal(
                first_carry.pose, held_frame.pose, residual_bone),
            "position/rotation residual setup snapped on Carry entry");
        require(
            vec_bits_equal(
                first_carry.pose.velocities[residual_bone],
                locomotion.velocities[residual_bone]) &&
                vec_bits_equal(
                    first_carry.pose.angular_velocities[residual_bone],
                    locomotion.angular_velocities[residual_bone]),
            "position/rotation residual setup introduced a velocity offset");

        const FlatWorldPose first_world = flat_world_pose(first_carry.pose);
        const FlatWorldPose target_world = flat_world_pose(locomotion);
        for (size_t bone = 0;
             bone < interaction::kFlatControllerBoneCount;
             ++bone) {
            require(
                length(
                    target_world.positions[bone] -
                    first_world.positions[bone]) <= 0.05F &&
                    rotation_distance(
                        target_world.rotations[bone],
                        first_world.rotations[bone]) <=
                        15.0F * 3.14159265358979323846F / 180.0F,
                "position/rotation residual setup exceeded the tight "
                "terminal step");
        }

        const ControllerInteractionFrameState deadline = handoff.apply(
            locomotion, output, 0.50F);
        if (rotational_residual) {
            require(
                !quat_bits_equal(
                    deadline.pose.rotations[residual_bone],
                    locomotion.rotations[residual_bone]),
                "nominal deadline discarded an isolated rotation offset");
        } else {
            require(
                !vec_bits_equal(
                    deadline.pose.positions[residual_bone],
                    locomotion.positions[residual_bone]),
                "nominal deadline discarded an isolated position offset");
        }

        bool converged = false;
        ControllerInteractionFrameState frame = deadline;
        for (int tick = 0; tick < 8 && !converged; ++tick) {
            frame = handoff.apply(
                locomotion, output, interaction::kControllerStepSeconds);
            converged = true;
            for (size_t bone = 0; bone <= 9U; ++bone) {
                converged = converged && flat_bone_channels_bits_equal(
                    frame.pose, locomotion, bone);
            }
        }
        require(
            converged,
            "position/rotation residual handoff did not converge "
            "bit-exactly");
    };

    run_residual_case(false);
    run_residual_case(true);
}

void test_layered_carry_terminal_accepts_tightly_bounded_moving_target() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::Hold;

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        entry, output, interaction::kControllerStepSeconds);

    FlatControllerPose locomotion = entry;
    locomotion.velocities[0].x = 0.50F;
    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = false;
    (void)handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    for (int tick = 0; tick < 12; ++tick) {
        locomotion.positions[0].x += 0.02F;
        (void)handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
    }

    locomotion.positions[0].x += 0.02F;
    ControllerInteractionFrameState frame = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    for (size_t bone = 0; bone <= 9U; ++bone) {
        require(
            flat_bone_channels_bits_equal(frame.pose, locomotion, bone),
            "terminal handoff rejected a tightly bounded moving target");
    }

    locomotion.positions[0].x += 0.02F;
    frame = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    for (size_t bone = 0; bone <= 9U; ++bone) {
        require(
            flat_bone_channels_bits_equal(frame.pose, locomotion, bone),
            "moving-target terminal handoff retained stale transition state");
    }
}

void test_layered_carry_keeps_fresh_lower_body_and_contacts_bit_exact() {
    FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.0F));
    RuntimeOutput output = make_owned_output(raw_reference);
    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(entry, output, interaction::kControllerStepSeconds);

    FlatControllerPose locomotion = entry;
    for (size_t bone = 0; bone <= 9U; ++bone) {
        const float value = static_cast<float>(bone + 1U);
        locomotion.positions[bone].x += 0.03F * value;
        locomotion.velocities[bone].z -= 0.04F * value;
        locomotion.rotations[bone] = quat_from_angle_axis(
            0.01F * value, vec3(0.0F, 1.0F, 0.0F));
        locomotion.angular_velocities[bone].y += 0.02F * value;
    }
    locomotion.foot_contacts = {0U, 1U};
    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = false;
    FlatControllerPose upper_target = entry;
    upper_target.rotations[12] = quat_from_angle_axis(
        0.15F, vec3(0.0F, 0.0F, 1.0F));
    output.pose = interaction::expand_flat_controller_pose(
        upper_target, raw_reference);
    output.pose.positions[8] = vec3(900.0F, -400.0F, 700.0F);
    output.pose.rotations[8] = quat_from_angle_axis(
        3.08F, vec3(1.0F, 0.0F, 0.0F));

    const auto require_layered_authority = [&](
        const ControllerInteractionFrameState& frame,
        const FlatControllerPose& fresh_locomotion) {
        for (size_t bone = 0; bone <= 9U; ++bone) {
            require(
                vec_bits_equal(
                    frame.pose.positions[bone],
                    fresh_locomotion.positions[bone]) &&
                    vec_bits_equal(
                        frame.pose.velocities[bone],
                        fresh_locomotion.velocities[bone]) &&
                    quat_bits_equal(
                        frame.pose.rotations[bone],
                        fresh_locomotion.rotations[bone]) &&
                    vec_bits_equal(
                        frame.pose.angular_velocities[bone],
                        fresh_locomotion.angular_velocities[bone]),
                "layered Carry changed fresh locomotion lower body");
        }
        require(
            frame.pose.foot_contacts == fresh_locomotion.foot_contacts,
            "layered Carry changed fresh locomotion contacts");
        require(
            !quat_bits_equal(
                frame.pose.rotations[12],
                fresh_locomotion.rotations[12]),
            "layered Carry failed to own the upper body");
    };

    ControllerInteractionFrameState frame = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    for (size_t bone = 0; bone <= 9U; ++bone) {
        require(
            flat_bone_channels_bits_equal(frame.pose, entry, bone),
            "first layered Carry frame did not preserve the rendered lower body");
    }
    require(
        frame.pose.foot_contacts == locomotion.foot_contacts,
        "first layered Carry frame did not publish fresh contacts");

    bool converged = false;
    for (int tick = 0; tick < 64 && !converged; ++tick) {
        frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        converged = true;
        for (size_t bone = 0; bone <= 9U; ++bone) {
            converged = converged && flat_bone_channels_bits_equal(
                frame.pose, locomotion, bone);
        }
    }
    require(converged, "layered Carry lower-body handoff did not converge");
    require_layered_authority(frame, locomotion);

    for (size_t bone = 0; bone <= 9U; ++bone) {
        const float value = static_cast<float>(bone + 1U);
        locomotion.positions[bone].z -= 0.005F * value;
        locomotion.velocities[bone].x += 0.006F * value;
    }
    locomotion.foot_contacts = {1U, 0U};
    frame = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    require_layered_authority(frame, locomotion);

    output.pose.positions[8] = vec3(-800.0F, 500.0F, -600.0F);
    output.pose.rotations[8] = quat_from_angle_axis(
        -3.02F, vec3(1.0F, 0.0F, 0.0F));
    frame = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    require_layered_authority(frame, locomotion);
}

void test_layered_carry_released_inactive_arm_uses_fresh_locomotion_authority() {
    constexpr std::array<size_t, 4> left_chain = {15U, 16U, 17U, 18U};
    constexpr std::array<size_t, 4> right_chain = {19U, 20U, 21U, 22U};

    for (const Hand active_hand : {Hand::Left, Hand::Right}) {
        const std::array<size_t, 4>& active_chain =
            active_hand == Hand::Left ? left_chain : right_chain;
        const std::array<size_t, 4>& inactive_chain =
            active_hand == Hand::Left ? right_chain : left_chain;

        const FlatControllerPose entry = make_flat_pose();
        const Pose raw_reference = interaction::expand_flat_controller_pose(
            entry, make_pose(0.375F));
        RuntimeOutput output = make_owned_output(raw_reference);
        output.diagnostics.hand = active_hand;

        ControllerInteractionFrameHandoff handoff;
        (void)handoff.apply(
            entry, output, interaction::kControllerStepSeconds);

        FlatControllerPose interaction_target = entry;
        interaction_target.rotations[12] = quat_from_angle_axis(
            0.52F, normalize(vec3(0.2F, 0.7F, 0.4F)));
        interaction_target.angular_velocities[12] =
            vec3(0.31F, -0.27F, 0.19F);
        for (size_t bone = 15U; bone <= 22U; ++bone) {
            const float value = static_cast<float>(bone + 1U);
            interaction_target.velocities[bone] =
                vec3(0.17F * value, -0.13F * value, 0.11F * value);
            interaction_target.rotations[bone] = quat_from_angle_axis(
                0.025F * value,
                normalize(vec3(0.3F, 0.5F + 0.01F * value, 0.7F)));
            interaction_target.angular_velocities[bone] =
                vec3(-0.09F * value, 0.07F * value, 0.05F * value);
        }
        output.pose = interaction::expand_flat_controller_pose(
            interaction_target, raw_reference);
        output.diagnostics.state = RuntimeState::Hold;
        (void)handoff.apply(
            entry, output, interaction::kControllerStepSeconds);
        output.diagnostics.state = RuntimeState::Carry;
        output.diagnostics.recorded_carry = false;
        output.diagnostics.inactive_arm_targets_locomotion = false;
        output.diagnostics.inactive_arm_tracks_locomotion = false;

        FlatControllerPose locomotion = entry;
        for (size_t bone = 10U; bone <= 22U; ++bone) {
            const float value = static_cast<float>(bone + 1U);
            locomotion.positions[bone] = locomotion.positions[bone] +
                vec3(0.003F * value, -0.002F * value, 0.004F * value);
            locomotion.velocities[bone] =
                vec3(-0.21F * value, 0.16F * value, -0.08F * value);
            locomotion.rotations[bone] = quat_from_angle_axis(
                0.018F * value,
                normalize(vec3(0.8F, 0.4F, 0.2F + 0.01F * value)));
            locomotion.angular_velocities[bone] =
                vec3(0.12F * value, -0.04F * value, 0.15F * value);
        }
        const ControllerInteractionFrameState before_release = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        output.diagnostics.inactive_arm_targets_locomotion = true;
        const FlatControllerPose locomotion_before = locomotion;
        const RuntimeOutput output_before = output;

        ControllerInteractionFrameState frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);

        for (size_t bone : inactive_chain) {
            require(
                flat_bone_channels_bits_equal(
                    frame.pose, before_release.pose, bone),
                "inactive Carry arm release was discontinuous on its first frame");
        }
        require(
            !flat_bone_channels_bits_equal(frame.pose, locomotion, 12U),
            "released inactive Carry arm surrendered spine authority");
        bool active_chain_remained_owned = false;
        for (size_t bone : active_chain) {
            active_chain_remained_owned = active_chain_remained_owned ||
                !flat_bone_channels_bits_equal(frame.pose, locomotion, bone);
        }
        require(
            active_chain_remained_owned,
            "released inactive Carry arm surrendered active-arm authority");
        require(
            flat_pose_bits_equal(locomotion, locomotion_before) &&
                output_fields_equal(output, output_before),
            "inactive-arm authority handoff mutated its inputs");

        const auto require_bounded_step = [&](const FlatControllerPose& previous) {
            const FlatWorldPose previous_world = flat_world_pose(previous);
            const FlatWorldPose current_world = flat_world_pose(frame.pose);
            for (size_t bone : inactive_chain) {
                require(
                    length(
                        current_world.positions[bone] -
                        previous_world.positions[bone]) <= 0.2F,
                    "inactive Carry arm blend exceeded the positional seam bound");
                require(
                    rotation_distance(
                        current_world.rotations[bone],
                        previous_world.rotations[bone]) <=
                        60.0F * 3.14159265358979323846F / 180.0F,
                    "inactive Carry arm blend exceeded the rotational seam bound");
            }
        };

        FlatControllerPose previous = frame.pose;
        for (int tick = 0; tick < 12; ++tick) {
            frame = handoff.apply(
                locomotion, output, interaction::kControllerStepSeconds);
            require_bounded_step(previous);
            previous = frame.pose;
        }
        bool still_blending = false;
        for (size_t bone : inactive_chain) {
            still_blending = still_blending ||
                !flat_bone_channels_bits_equal(
                    frame.pose, locomotion, bone);
        }
        require(
            still_blending,
            "inactive Carry arm release ended before 0.50 seconds at 25 Hz");

        frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        require_bounded_step(previous);
        for (size_t bone : inactive_chain) {
            require(
                flat_bone_channels_bits_equal(
                    frame.pose, locomotion, bone),
                "inactive Carry arm did not converge within one 25 Hz tick of 0.50 seconds");
        }
    }
}

void test_inactive_arm_locomotion_intent_does_not_change_other_carry_modes() {
    constexpr std::array<size_t, 4> inactive_left = {15U, 16U, 17U, 18U};
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));

    FlatControllerPose interaction_target = entry;
    for (size_t bone : inactive_left) {
        interaction_target.rotations[bone] = quat_from_angle_axis(
            0.65F,
            normalize(vec3(
                0.2F + 0.01F * static_cast<float>(bone),
                0.8F,
                0.3F)));
    }
    FlatControllerPose locomotion = entry;
    for (size_t bone : inactive_left) {
        locomotion.positions[bone].x += 0.05F;
        locomotion.velocities[bone].y -= 0.07F;
        locomotion.rotations[bone] = quat_from_angle_axis(
            0.12F, vec3(0.0F, 1.0F, 0.0F));
        locomotion.angular_velocities[bone].z += 0.09F;
    }

    const auto render = [&](
        bool recorded_carry,
        bool inactive_arm_tracks_locomotion,
        bool inactive_arm_targets_locomotion) {
        RuntimeOutput output = make_owned_output(raw_reference);
        output.diagnostics.hand = Hand::Right;
        ControllerInteractionFrameHandoff handoff;
        (void)handoff.apply(
            entry, output, interaction::kControllerStepSeconds);
        output.pose = interaction::expand_flat_controller_pose(
            interaction_target, raw_reference);
        output.diagnostics.state = RuntimeState::Hold;
        (void)handoff.apply(
            entry, output, interaction::kControllerStepSeconds);
        output.diagnostics.state = RuntimeState::Carry;
        output.diagnostics.recorded_carry = recorded_carry;
        output.diagnostics.inactive_arm_targets_locomotion =
            inactive_arm_targets_locomotion;
        output.diagnostics.inactive_arm_tracks_locomotion =
            inactive_arm_tracks_locomotion;
        return handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
    };

    const ControllerInteractionFrameState unreleased =
        render(false, false, false);
    const ControllerInteractionFrameState exact_flag_only =
        render(false, true, false);
    const ControllerInteractionFrameState recorded = render(true, true, true);
    for (size_t bone : inactive_left) {
        require(
            !flat_bone_channels_bits_equal(
                unreleased.pose, locomotion, bone),
            "false inactive-arm intent unexpectedly surrendered authority");
        require(
            !flat_bone_channels_bits_equal(
                exact_flag_only.pose, locomotion, bone),
            "exact-tracking flag was reinterpreted as release intent");
        require(
            !flat_bone_channels_bits_equal(recorded.pose, locomotion, bone),
            "recorded Carry unexpectedly surrendered inactive-arm authority");
    }
    require(
        !unreleased.synchronize_simulation_root &&
            !exact_flag_only.synchronize_simulation_root &&
            recorded.synchronize_simulation_root,
        "inactive-arm intent changed Carry root authority policy");
}

void test_inactive_arm_release_bounds_moving_target_and_spine() {
    constexpr std::array<size_t, 4> left_chain = {15U, 16U, 17U, 18U};
    constexpr std::array<size_t, 4> right_chain = {19U, 20U, 21U, 22U};
    constexpr float rotation_step_limit =
        60.0F * 3.14159265358979323846F / 180.0F;

    for (const Hand active_hand : {Hand::Left, Hand::Right}) {
        const std::array<size_t, 4>& active_chain =
            active_hand == Hand::Left ? left_chain : right_chain;
        const std::array<size_t, 4>& inactive_chain =
            active_hand == Hand::Left ? right_chain : left_chain;
        const FlatControllerPose entry = make_flat_pose();
        const Pose raw_reference = interaction::expand_flat_controller_pose(
            entry, make_pose(0.375F));
        FlatControllerPose interaction_target = entry;
        interaction_target.rotations[12] = quat_from_angle_axis(
            2.80F, normalize(vec3(0.2F, 0.8F, 0.3F)));
        for (size_t bone : active_chain) {
            interaction_target.rotations[bone] = quat_from_angle_axis(
                0.85F,
                normalize(vec3(0.3F, 0.6F, 0.5F)));
        }
        for (size_t bone : inactive_chain) {
            interaction_target.rotations[bone] = quat_from_angle_axis(
                2.98F,
                normalize(vec3(
                    0.2F + 0.01F * static_cast<float>(bone),
                    0.7F,
                    0.4F)));
        }

        RuntimeOutput output = make_owned_output(raw_reference);
        output.diagnostics.hand = active_hand;
        ControllerInteractionFrameHandoff handoff;
        (void)handoff.apply(
            entry, output, interaction::kControllerStepSeconds);
        output.pose = interaction::expand_flat_controller_pose(
            interaction_target, raw_reference);
        output.diagnostics.state = RuntimeState::Hold;
        (void)handoff.apply(
            entry, output, interaction::kControllerStepSeconds);
        output.diagnostics.state = RuntimeState::Carry;
        output.diagnostics.recorded_carry = false;
        output.diagnostics.inactive_arm_targets_locomotion = false;

        FlatControllerPose locomotion = entry;
        const ControllerInteractionFrameState before_release = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        interaction_target.rotations[12] = quat_from_angle_axis(
            2.80F, normalize(vec3(0.2F, 0.8F, 0.3F)));
        output.pose = interaction::expand_flat_controller_pose(
            interaction_target, raw_reference);
        output.diagnostics.inactive_arm_targets_locomotion = true;
        ControllerInteractionFrameState frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        const FlatWorldPose before_release_world =
            flat_world_pose(before_release.pose);
        const FlatWorldPose first_release_world = flat_world_pose(frame.pose);
        for (size_t bone : inactive_chain) {
            require(
                length(
                    first_release_world.positions[bone] -
                    before_release_world.positions[bone]) <= 0.2F,
                "parent-only inactive arm release exceeded positional bound");
            require(
                rotation_distance(
                    first_release_world.rotations[bone],
                    before_release_world.rotations[bone]) <=
                    rotation_step_limit,
                "parent-only inactive arm release exceeded rotational bound");
        }

        FlatControllerPose previous = frame.pose;
        for (int tick = 0; tick < 80; ++tick) {
            if (tick < 10) {
                const float progress = static_cast<float>(tick + 1);
                interaction_target.rotations[12] = quat_from_angle_axis(
                    2.80F + 0.025F * progress,
                    normalize(vec3(0.2F, 0.8F, 0.3F)));
                for (size_t bone : inactive_chain) {
                    locomotion.positions[bone].x += 0.0005F;
                    locomotion.rotations[bone] = quat_from_angle_axis(
                        0.012F * progress,
                        normalize(vec3(
                            0.8F,
                            0.3F + 0.01F * static_cast<float>(bone),
                            0.4F)));
                }
                output.pose = interaction::expand_flat_controller_pose(
                    interaction_target, raw_reference);
            }
            const FlatControllerPose locomotion_before = locomotion;
            const RuntimeOutput output_before = output;
            frame = handoff.apply(
                locomotion, output, interaction::kControllerStepSeconds);
            require(
                flat_pose_bits_equal(locomotion, locomotion_before) &&
                    output_fields_equal(output, output_before),
                "bounded inactive-arm handoff mutated moving inputs");
            const FlatWorldPose previous_world = flat_world_pose(previous);
            const FlatWorldPose current_world = flat_world_pose(frame.pose);
            for (size_t bone : inactive_chain) {
                require(
                    length(
                        current_world.positions[bone] -
                        previous_world.positions[bone]) <= 0.2F,
                    "moving-target inactive arm exceeded positional bound");
                require(
                    rotation_distance(
                        current_world.rotations[bone],
                        previous_world.rotations[bone]) <=
                        rotation_step_limit,
                    "moving-target inactive arm exceeded rotational bound");
            }
            previous = frame.pose;
        }

        for (size_t bone : inactive_chain) {
            require(
                flat_bone_channels_bits_equal(
                    frame.pose, locomotion, bone),
                "bounded inactive arm did not converge to moving locomotion target");
        }
        require(
            !flat_bone_channels_bits_equal(frame.pose, locomotion, 12U),
            "bounded inactive arm surrendered moving spine authority");
        bool active_chain_owned = false;
        for (size_t bone : active_chain) {
            active_chain_owned = active_chain_owned ||
                !flat_bone_channels_bits_equal(
                    frame.pose, locomotion, bone);
        }
        require(
            active_chain_owned,
            "bounded inactive arm surrendered active-arm authority");

        previous = frame.pose;
        for (size_t bone : inactive_chain) {
            locomotion.positions[bone].z += 0.0002F;
            locomotion.rotations[bone] = quat_mul(
                quat_from_angle_axis(
                    0.005F,
                    normalize(vec3(0.4F, 0.7F, 0.2F))),
                locomotion.rotations[bone]);
        }
        frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        const FlatWorldPose tracked_previous_world = flat_world_pose(previous);
        const FlatWorldPose tracked_world = flat_world_pose(frame.pose);
        for (size_t bone : inactive_chain) {
            require(
                flat_bone_channels_bits_equal(
                    frame.pose, locomotion, bone),
                "converged inactive arm did not track locomotion in the same tick");
            require(
                length(
                    tracked_world.positions[bone] -
                    tracked_previous_world.positions[bone]) <= 0.2F &&
                    rotation_distance(
                        tracked_world.rotations[bone],
                        tracked_previous_world.rotations[bone]) <=
                        rotation_step_limit,
                "same-tick inactive arm tracking exceeded its bounds");
        }
    }
}

void test_inactive_arm_return_to_recorded_target_is_bounded_and_convergent() {
    constexpr std::array<size_t, 4> left_chain = {15U, 16U, 17U, 18U};
    constexpr std::array<size_t, 4> right_chain = {19U, 20U, 21U, 22U};
    constexpr float rotation_step_limit =
        60.0F * 3.14159265358979323846F / 180.0F;

    for (const Hand active_hand : {Hand::Left, Hand::Right}) {
        const std::array<size_t, 4>& active_chain =
            active_hand == Hand::Left ? left_chain : right_chain;
        const std::array<size_t, 4>& inactive_chain =
            active_hand == Hand::Left ? right_chain : left_chain;
        const FlatControllerPose entry = make_flat_pose();
        const Pose raw_reference = interaction::expand_flat_controller_pose(
            entry, make_pose(0.375F));
        FlatControllerPose layered_target = entry;
        for (size_t bone : active_chain) {
            layered_target.rotations[bone] = quat_from_angle_axis(
                0.70F, normalize(vec3(0.2F, 0.6F, 0.5F)));
        }
        for (size_t bone : inactive_chain) {
            layered_target.rotations[bone] = quat_from_angle_axis(
                2.60F, normalize(vec3(0.3F, 0.7F, 0.4F)));
        }

        RuntimeOutput output = make_owned_output(raw_reference);
        output.diagnostics.hand = active_hand;
        ControllerInteractionFrameHandoff handoff;
        (void)handoff.apply(
            entry, output, interaction::kControllerStepSeconds);
        output.pose = interaction::expand_flat_controller_pose(
            layered_target, raw_reference);
        output.diagnostics.state = RuntimeState::Hold;
        (void)handoff.apply(
            entry, output, interaction::kControllerStepSeconds);
        output.diagnostics.state = RuntimeState::Carry;
        output.diagnostics.recorded_carry = false;
        output.diagnostics.inactive_arm_targets_locomotion = false;

        FlatControllerPose locomotion = entry;
        (void)handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        output.diagnostics.inactive_arm_targets_locomotion = true;
        ControllerInteractionFrameState frame{};
        for (int tick = 0; tick < 30; ++tick) {
            frame = handoff.apply(
                locomotion, output, interaction::kControllerStepSeconds);
        }
        for (size_t bone : inactive_chain) {
            require(
                flat_bone_channels_bits_equal(
                    frame.pose, locomotion, bone),
                "inactive arm did not settle to locomotion before recorded switch");
        }

        FlatControllerPose recorded_target = entry;
        recorded_target.rotations[12] = quat_from_angle_axis(
            2.70F, normalize(vec3(0.2F, 0.8F, 0.3F)));
        for (size_t bone : active_chain) {
            recorded_target.rotations[bone] = quat_from_angle_axis(
                0.95F, normalize(vec3(0.4F, 0.5F, 0.7F)));
        }
        for (size_t bone : inactive_chain) {
            recorded_target.positions[bone].x += 0.02F;
            recorded_target.velocities[bone] =
                vec3(0.7F, -0.5F, 0.3F);
            recorded_target.rotations[bone] = quat_from_angle_axis(
                2.85F,
                normalize(vec3(
                    0.3F,
                    0.5F + 0.01F * static_cast<float>(bone),
                    0.8F)));
            recorded_target.angular_velocities[bone] =
                vec3(-0.6F, 0.4F, 0.8F);
        }
        output.pose = interaction::expand_flat_controller_pose(
            recorded_target, raw_reference);
        output.diagnostics.recorded_carry = true;
        output.diagnostics.inactive_arm_targets_locomotion = false;

        FlatControllerPose previous = frame.pose;
        frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        const auto require_bounded_step = [&](const FlatControllerPose& prior) {
            const FlatWorldPose prior_world = flat_world_pose(prior);
            const FlatWorldPose current_world = flat_world_pose(frame.pose);
            for (size_t bone : inactive_chain) {
                require(
                    length(
                        current_world.positions[bone] -
                        prior_world.positions[bone]) <= 0.2F,
                    "recorded return exceeded inactive-arm positional bound");
                require(
                    rotation_distance(
                        current_world.rotations[bone],
                        prior_world.rotations[bone]) <=
                        rotation_step_limit,
                    "recorded return exceeded inactive-arm rotational bound");
            }
        };
        require_bounded_step(previous);
        previous = frame.pose;

        for (int tick = 0; tick < 80; ++tick) {
            if (tick < 10) {
                const float progress = static_cast<float>(tick + 1);
                recorded_target.rotations[12] = quat_from_angle_axis(
                    2.70F + 0.01F * progress,
                    normalize(vec3(0.2F, 0.8F, 0.3F)));
                for (size_t bone : inactive_chain) {
                    recorded_target.positions[bone].z += 0.0003F;
                    recorded_target.rotations[bone] = quat_from_angle_axis(
                        2.85F - 0.01F * progress,
                        normalize(vec3(
                            0.3F,
                            0.5F + 0.01F * static_cast<float>(bone),
                            0.8F)));
                }
                output.pose = interaction::expand_flat_controller_pose(
                    recorded_target, raw_reference);
            }
            frame = handoff.apply(
                locomotion, output, interaction::kControllerStepSeconds);
            require_bounded_step(previous);
            previous = frame.pose;
        }

        const FlatControllerPose expected = interaction::collapse_interaction_pose(
            output.pose, raw_reference, entry);
        for (size_t bone : inactive_chain) {
            require(
                flat_bone_channels_bits_equal(
                    frame.pose, expected, bone),
                "inactive arm did not converge to recorded interaction target");
        }
        require(
            !flat_bone_channels_bits_equal(frame.pose, locomotion, 12U),
            "recorded inactive-arm return surrendered spine authority");
    }
}

void test_frame_handoff_release_is_continuous_and_relinquishes_after_blend() {
    const FlatControllerPose entry = make_flat_pose();
    FlatControllerPose target = entry;
    target.positions[0] = target.positions[0] + vec3(1.0F, 0.5F, -2.0F);
    target.rotations[2] = quat_from_angle_axis(
        0.9F, normalize(vec3(0.4F, 0.8F, 0.1F)));
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.375F));
    RuntimeOutput owned = make_owned_output(raw_reference);
    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        entry, owned, interaction::kControllerStepSeconds);
    owned.pose = interaction::expand_flat_controller_pose(
        target, raw_reference);
    const ControllerInteractionFrameState settled = handoff.apply(
        entry, owned, interaction::kControllerStepSeconds);
    require_flat_pose_near(
        settled.pose, target, "owned setup did not settle on target");

    FlatControllerPose fresh_locomotion = entry;
    fresh_locomotion.positions[0] =
        fresh_locomotion.positions[0] + vec3(-3.0F, 0.0F, 4.0F);
    RuntimeOutput idle;
    const ControllerInteractionFrameState first_release = handoff.apply(
        fresh_locomotion, idle, interaction::kControllerStepSeconds);
    require(!first_release.runtime_owns_pose, "release retained runtime ownership");
    require(
        first_release.overrides_locomotion_pose,
        "release dropped visual override on its first frame");
    require_flat_pose_near(
        first_release.pose,
        settled.pose,
        "first release frame was discontinuous");

    bool relinquished = false;
    for (int tick = 0; tick < 30; ++tick) {
        const ControllerInteractionFrameState frame = handoff.apply(
            fresh_locomotion, idle, interaction::kControllerStepSeconds);
        require(!frame.runtime_owns_pose, "release regained runtime ownership");
        if (!frame.overrides_locomotion_pose) {
            require(
                flat_pose_bits_equal(frame.pose, fresh_locomotion),
                "release relinquished before converging to fresh locomotion");
            relinquished = true;
            break;
        }
    }
    require(relinquished, "release never relinquished visual override");
}

void test_all_place_states_keep_one_full_body_runtime_ownership_epoch() {
    const FlatControllerPose locomotion = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        locomotion, make_pose(0.625F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = false;
    output.diagnostics.target = {80U, 3U};
    output.diagnostics.object_state = ObjectState::Held;
    output.diagnostics.attached = true;

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);

    FlatControllerPose place_pose = locomotion;
    place_pose.positions[0] =
        place_pose.positions[0] + vec3(0.65F, 0.10F, -0.35F);
    place_pose.rotations[6] = quat_from_angle_axis(
        0.55F, normalize(vec3(0.2F, 0.8F, 0.3F)));
    place_pose.rotations[12] = quat_from_angle_axis(
        0.35F, normalize(vec3(0.4F, 0.5F, 0.7F)));

    const std::array<RuntimeState, 4> place_states = {
        RuntimeState::PlacePreflight,
        RuntimeState::PlaceAlign,
        RuntimeState::PlaceReplay,
        RuntimeState::PlaceRelease};
    for (size_t index = 0; index < place_states.size(); ++index) {
        place_pose.positions[0].x = place_pose.positions[0].x +
            0.02F * static_cast<float>(index);
        output.pose = interaction::expand_flat_controller_pose(
            place_pose, raw_reference);
        output.diagnostics.state = place_states[index];
        if (place_states[index] == RuntimeState::PlaceRelease) {
            output.diagnostics.target = {80U, 4U};
            output.diagnostics.object_state = ObjectState::Free;
            output.diagnostics.attached = false;
        }
        const ControllerInteractionFrameState frame = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        require(
            frame.runtime_owns_pose && frame.overrides_locomotion_pose,
            "place state ended the runtime ownership epoch");
        require(
            frame.synchronize_simulation_root,
            "full-body place state was treated as layered Carry");
        require_flat_pose_near(
            frame.pose,
            place_pose,
            "place state did not publish its full-body runtime pose");
        require(
            !flat_bone_channels_bits_equal(frame.pose, locomotion, 0U) &&
                !flat_bone_channels_bits_equal(frame.pose, locomotion, 6U),
            "place state replaced runtime root/legs with locomotion");
    }
}

void test_place_release_generation_change_ends_constraint_without_free_object_solve() {
    const FlatControllerPose locomotion = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        locomotion, make_pose(0.75F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::PlaceAlign;
    output.diagnostics.target = {81U, 6U};
    output.diagnostics.affordance_id = 12U;
    output.diagnostics.hand = Hand::Right;
    output.diagnostics.object_state = ObjectState::Held;
    output.diagnostics.attached = true;
    output.diagnostics.hand_constraint_weight = 0.0F;
    const ControllerInteractionHandConstraint held_constraint =
        make_hand_constraint(output, locomotion, Hand::Right);

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        held_constraint);
    output.diagnostics.hand_constraint_weight = 1.0F;
    const ControllerInteractionFrameState held = handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        held_constraint);
    require(
        held.hand_constraint_result.applied,
        "held setup did not activate the semantic hand constraint");
    const FlatWorldPose held_world = flat_world_pose(held.pose);

    output.diagnostics.state = RuntimeState::PlaceRelease;
    output.diagnostics.target = {81U, 7U};
    output.diagnostics.object_state = ObjectState::Free;
    output.diagnostics.attached = false;
    output.diagnostics.hand_constraint_weight = 0.0F;
    FlatControllerPose authored_release = locomotion;
    authored_release.positions[0] = authored_release.positions[0] +
        vec3(0.04F, 0.01F, -0.03F);
    authored_release.velocities[0] = authored_release.velocities[0] +
        vec3(0.10F, -0.02F, 0.06F);
    authored_release.rotations[0] = quat_normalize(quat_mul(
        quat_from_angle_axis(
            0.08F, normalize(vec3(0.2F, 0.9F, 0.3F))),
        locomotion.rotations[0]));
    authored_release.angular_velocities[0] =
        authored_release.angular_velocities[0] +
        vec3(0.02F, -0.03F, 0.04F);
    authored_release.rotations[12] = quat_normalize(quat_mul(
        quat_from_angle_axis(
            0.12F, normalize(vec3(0.3F, 0.4F, 0.8F))),
        locomotion.rotations[12]));
    authored_release.angular_velocities[12] =
        authored_release.angular_velocities[12] +
        vec3(-0.05F, 0.04F, 0.03F);
    authored_release.rotations[15] = quat_normalize(quat_mul(
        quat_from_angle_axis(
            0.16F, normalize(vec3(0.7F, 0.2F, 0.5F))),
        locomotion.rotations[15]));
    authored_release.velocities[17] = authored_release.velocities[17] +
        vec3(0.03F, -0.02F, 0.01F);
    authored_release.foot_contacts = {
        static_cast<uint8_t>(1U - locomotion.foot_contacts[0]),
        static_cast<uint8_t>(1U - locomotion.foot_contacts[1])};
    output.pose = interaction::expand_flat_controller_pose(
        authored_release, raw_reference);
    FlatControllerPose expected_authored_release =
        interaction::collapse_interaction_pose(
            output.pose, raw_reference, locomotion);
    for (size_t bone = 0U;
         bone < expected_authored_release.rotations.size();
         ++bone) {
        if (quat_dot(
                held.pose.rotations[bone],
                expected_authored_release.rotations[bone]) < 0.0F) {
            expected_authored_release.rotations[bone] =
                -expected_authored_release.rotations[bone];
        }
    }
    ControllerInteractionHandConstraint free_constraint = held_constraint;
    free_constraint.target = output.diagnostics.target;
    free_constraint.grasp_world.position =
        free_constraint.grasp_world.position +
        vec3(0.30F, 0.10F, -0.20F);
    const ControllerInteractionFrameState released = handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        free_constraint);
    require(
        released.runtime_owns_pose,
        "atomic object release incorrectly ended pose ownership");
    require(
        !released.hand_constraint_validated &&
            !released.hand_constraint_result.applied,
        "new free target generation received an IK solve");
    const FlatWorldPose released_world = flat_world_pose(released.pose);
    float maximum_active_arm_translation_m = 0.0F;
    float maximum_active_arm_velocity_mps = 0.0F;
    float maximum_active_arm_rotation_radians = 0.0F;
    float maximum_active_arm_angular_velocity_radians_per_second = 0.0F;
    for (size_t bone = 19U; bone <= 22U; ++bone) {
        maximum_active_arm_translation_m = std::max(
            maximum_active_arm_translation_m,
            length(
                released_world.positions[bone] -
                held_world.positions[bone]));
        maximum_active_arm_velocity_mps = std::max(
            maximum_active_arm_velocity_mps,
            length(
                released_world.velocities[bone] -
                held_world.velocities[bone]));
        maximum_active_arm_rotation_radians = std::max(
            maximum_active_arm_rotation_radians,
            rotation_distance(
                released_world.rotations[bone],
                held_world.rotations[bone]));
        maximum_active_arm_angular_velocity_radians_per_second = std::max(
            maximum_active_arm_angular_velocity_radians_per_second,
            length(
                released_world.angular_velocities[bone] -
                held_world.angular_velocities[bone]));
    }
    if (maximum_active_arm_translation_m > 1.0e-5F ||
        maximum_active_arm_velocity_mps > 1.0e-5F ||
        maximum_active_arm_rotation_radians > 1.0e-5F ||
        maximum_active_arm_angular_velocity_radians_per_second > 1.0e-5F) {
        throw std::runtime_error(
            "first PlaceRelease active-arm step was " +
            std::to_string(maximum_active_arm_translation_m) +
            " m / " + std::to_string(maximum_active_arm_velocity_mps) +
            " mps / " +
            std::to_string(maximum_active_arm_rotation_radians) +
            " radians / " +
            std::to_string(
                maximum_active_arm_angular_velocity_radians_per_second) +
            " radps");
    }
    for (size_t bone = 0U; bone < released.pose.positions.size(); ++bone) {
        if (bone >= 19U && bone <= 22U) continue;
        if (!flat_bone_channels_bits_equal(
                released.pose, expected_authored_release, bone)) {
            const char* channel =
                !vec_bits_equal(
                    released.pose.positions[bone],
                    expected_authored_release.positions[bone])
                ? "position"
                : (!vec_bits_equal(
                       released.pose.velocities[bone],
                       expected_authored_release.velocities[bone])
                       ? "velocity"
                       : (!quat_bits_equal(
                              released.pose.rotations[bone],
                              expected_authored_release.rotations[bone])
                              ? "rotation"
                              : "angular velocity"));
            throw std::runtime_error(
                "active-arm release preservation changed authored bone " +
                std::to_string(bone) + " " + channel);
        }
    }
    require(
        released.pose.foot_contacts == expected_authored_release.foot_contacts,
        "active-arm release preservation changed authored foot contacts");

    FlatControllerPose previous_displayed = released.pose;
    float release_elapsed_seconds = interaction::kControllerStepSeconds;
    bool reached_authored_retract = false;
    for (int tick = 1; tick <= 10; ++tick) {
        const float progress = static_cast<float>(tick);
        FlatControllerPose authored_retract = authored_release;
        authored_retract.positions[0] = authored_retract.positions[0] +
            vec3(0.002F * progress, 0.0F, -0.001F * progress);
        authored_retract.velocities[0] = authored_retract.velocities[0] +
            vec3(0.003F * progress, 0.0F, 0.002F * progress);
        authored_retract.rotations[12] = quat_normalize(quat_mul(
            quat_from_angle_axis(
                0.01F * progress,
                normalize(vec3(0.4F, 0.3F, 0.8F))),
            authored_release.rotations[12]));
        authored_retract.rotations[15] = quat_normalize(quat_mul(
            quat_from_angle_axis(
                -0.012F * progress,
                normalize(vec3(0.6F, 0.5F, 0.2F))),
            authored_release.rotations[15]));
        authored_retract.rotations[20] = quat_normalize(quat_mul(
            quat_from_angle_axis(
                0.015F * progress,
                normalize(vec3(0.2F, 0.7F, 0.4F))),
            authored_release.rotations[20]));
        authored_retract.angular_velocities[21] =
            authored_retract.angular_velocities[21] +
            vec3(0.01F * progress, -0.008F * progress, 0.006F * progress);
        authored_retract.foot_contacts = {
            static_cast<uint8_t>(tick % 2),
            static_cast<uint8_t>((tick + 1) % 2)};
        output.pose = interaction::expand_flat_controller_pose(
            authored_retract, raw_reference);
        FlatControllerPose expected_authored_retract =
            interaction::collapse_interaction_pose(
                output.pose, raw_reference, locomotion);
        for (size_t bone = 0U;
             bone < expected_authored_retract.rotations.size();
             ++bone) {
            if (quat_dot(
                    previous_displayed.rotations[bone],
                    expected_authored_retract.rotations[bone]) < 0.0F) {
                expected_authored_retract.rotations[bone] =
                    -expected_authored_retract.rotations[bone];
            }
        }

        const ControllerInteractionFrameState retract = handoff.apply(
            locomotion,
            output,
            interaction::kControllerStepSeconds,
            free_constraint);
        require(
            retract.runtime_owns_pose &&
                !retract.hand_constraint_validated &&
                !retract.hand_constraint_result.applied,
            "runtime-owned Free retract solved a hand constraint");
        for (size_t bone = 0U; bone < retract.pose.positions.size(); ++bone) {
            if (bone >= 19U && bone <= 22U) continue;
            require(
                flat_bone_channels_bits_equal(
                    retract.pose, expected_authored_retract, bone),
                "active-arm retract blend changed a non-active authored channel");
        }
        require(
            retract.pose.foot_contacts ==
                expected_authored_retract.foot_contacts,
            "active-arm retract blend changed authored foot contacts");

        const FlatWorldPose previous_world =
            flat_world_pose(previous_displayed);
        const FlatWorldPose retract_world = flat_world_pose(retract.pose);
        float maximum_translation_step_m = 0.0F;
        float maximum_rotation_step_radians = 0.0F;
        for (size_t bone = 19U; bone <= 22U; ++bone) {
            maximum_translation_step_m = std::max(
                maximum_translation_step_m,
                length(
                    retract_world.positions[bone] -
                    previous_world.positions[bone]));
            maximum_rotation_step_radians = std::max(
                maximum_rotation_step_radians,
                rotation_distance(
                    retract_world.rotations[bone],
                    previous_world.rotations[bone]));
        }
        require(
            maximum_translation_step_m <= 0.20F + 1.0e-5F &&
                maximum_rotation_step_radians <= 1.047197551F + 1.0e-5F,
            "active-arm retract blend exceeded one 25 Hz bounded step");

        bool selected_arm_is_authored = true;
        for (size_t bone = 19U; bone <= 22U; ++bone) {
            selected_arm_is_authored = selected_arm_is_authored &&
                flat_bone_channels_bits_equal(
                    retract.pose, expected_authored_retract, bone);
        }
        if (release_elapsed_seconds < 0.25F) {
            require(
                !selected_arm_is_authored,
                "active-arm retract blend ended before 0.25 seconds");
        } else {
            require(
                selected_arm_is_authored,
                "active-arm retract blend missed its authored endpoint");
            reached_authored_retract = true;
        }
        previous_displayed = retract.pose;
        if (reached_authored_retract) break;
        release_elapsed_seconds += interaction::kControllerStepSeconds;
    }
    require(
        reached_authored_retract &&
            release_elapsed_seconds >= 0.25F &&
            release_elapsed_seconds <=
                0.25F + interaction::kControllerStepSeconds,
        "active-arm retract blend escaped one 25 Hz tick of 0.25 seconds");

    output.diagnostics.target = held_constraint.target;
    output.diagnostics.object_state = ObjectState::Held;
    output.diagnostics.attached = true;
    const ControllerInteractionFrameState stale_generation = handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        held_constraint);
    require(
        !stale_generation.hand_constraint_validated &&
            !stale_generation.hand_constraint_result.applied,
        "ended hand-constraint epoch resurrected after generation rollback");
}

void test_normal_release_starts_at_last_displayed_place_release_pose() {
    const FlatControllerPose locomotion = make_flat_pose();
    FlatControllerPose place_release_pose = locomotion;
    place_release_pose.positions[0] =
        place_release_pose.positions[0] + vec3(0.45F, 0.20F, -0.30F);
    place_release_pose.rotations[12] = quat_from_angle_axis(
        0.70F, normalize(vec3(0.2F, 0.6F, 0.7F)));
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        locomotion, make_pose(0.875F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::PlaceRelease;
    output.diagnostics.target = {82U, 9U};
    output.diagnostics.object_state = ObjectState::Free;

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    output.pose = interaction::expand_flat_controller_pose(
        place_release_pose, raw_reference);
    const ControllerInteractionFrameState displayed = handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    require_flat_pose_near(
        displayed.pose,
        place_release_pose,
        "PlaceRelease setup did not publish its final pose");

    FlatControllerPose fresh_locomotion = locomotion;
    fresh_locomotion.positions[0] =
        fresh_locomotion.positions[0] + vec3(-2.0F, 0.0F, 1.5F);
    const ControllerInteractionFrameState first_release = handoff.apply(
        fresh_locomotion,
        RuntimeOutput{},
        interaction::kControllerStepSeconds);
    require(
        !first_release.runtime_owns_pose &&
            first_release.overrides_locomotion_pose &&
            flat_pose_bits_equal(first_release.pose, displayed.pose),
        "0.25 second release did not start at the last PlaceRelease pose");
}

void test_place_release_active_arm_return_is_bounded_after_ownership() {
    constexpr float kMaximumTranslationStepM = 0.04F;
    constexpr float kMaximumRotationStepRadians =
        10.0F * 3.14159265358979323846F / 180.0F;

    for (Hand hand : {Hand::Left, Hand::Right}) {
        const FlatControllerPose locomotion = make_flat_pose();
        FlatControllerPose final_place_release = locomotion;
        const size_t arm_begin = hand == Hand::Left ? 15U : 19U;
        const std::array<float, 4> return_angles = {
            0.35F, -0.25F, 0.20F, 1.40F};
        const std::array<vec3, 4> return_axes = {
            normalize(vec3(0.3F, 0.8F, 0.4F)),
            normalize(vec3(0.7F, 0.2F, 0.5F)),
            normalize(vec3(0.2F, 0.6F, 0.8F)),
            normalize(vec3(0.5F, 0.7F, 0.3F))};
        for (size_t offset = 0U; offset < 4U; ++offset) {
            const size_t bone = arm_begin + offset;
            final_place_release.rotations[bone] = quat_normalize(quat_mul(
                quat_from_angle_axis(
                    return_angles[offset], return_axes[offset]),
                locomotion.rotations[bone]));
        }

        const Pose raw_reference = interaction::expand_flat_controller_pose(
            locomotion, make_pose(0.90625F));
        RuntimeOutput output = make_owned_output(raw_reference);
        output.diagnostics.state = RuntimeState::PlaceRelease;
        output.diagnostics.target = {
            hand == Hand::Left ? 90U : 91U, 1U};
        output.diagnostics.affordance_id = 17U;
        output.diagnostics.hand = hand;
        output.diagnostics.object_state = ObjectState::Free;
        output.diagnostics.attached = false;

        ControllerInteractionFrameHandoff handoff;
        (void)handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        output.pose = interaction::expand_flat_controller_pose(
            final_place_release, raw_reference, locomotion);
        const ControllerInteractionFrameState displayed = handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        require_flat_pose_near(
            displayed.pose,
            final_place_release,
            "PlaceRelease active-arm return setup did not reach its pose");

        RuntimeOutput idle;
        ControllerInteractionFrameState previous = handoff.apply(
            locomotion, idle, interaction::kControllerStepSeconds);
        require(
            previous.overrides_locomotion_pose &&
                flat_pose_bits_equal(previous.pose, displayed.pose),
            "post-PlaceRelease return was discontinuous on its first Free frame");

        bool relinquished = false;
        int release_tick = 0;
        for (int tick = 1; tick <= 30; ++tick) {
            const ControllerInteractionFrameState frame = handoff.apply(
                locomotion, idle, interaction::kControllerStepSeconds);
            const FlatWorldPose previous_world =
                flat_world_pose(previous.pose);
            const FlatWorldPose frame_world = flat_world_pose(frame.pose);
            float maximum_translation_step_m = 0.0F;
            float maximum_rotation_step_radians = 0.0F;
            for (size_t bone = arm_begin; bone < arm_begin + 4U; ++bone) {
                maximum_translation_step_m = std::max(
                    maximum_translation_step_m,
                    length(
                        frame_world.positions[bone] -
                        previous_world.positions[bone]));
                maximum_rotation_step_radians = std::max(
                    maximum_rotation_step_radians,
                    rotation_distance(
                        previous_world.rotations[bone],
                        frame_world.rotations[bone]));
            }
            if (maximum_translation_step_m >
                    kMaximumTranslationStepM + 1.0e-5F ||
                maximum_rotation_step_radians >
                    kMaximumRotationStepRadians + 1.0e-5F) {
                throw std::runtime_error(
                    "post-PlaceRelease active-arm return step was " +
                    std::to_string(maximum_translation_step_m) + " m / " +
                    std::to_string(
                        maximum_rotation_step_radians * 180.0F /
                        3.14159265358979323846F) +
                    " degrees");
            }

            if (tick >= 7) {
                for (size_t bone = 0U;
                     bone < interaction::kFlatControllerBoneCount;
                    ++bone) {
                    if (bone >= arm_begin && bone < arm_begin + 4U) continue;
                    require(
                        flat_bone_channels_bits_equal(
                            frame.pose, locomotion, bone),
                        "active-arm return changed a non-arm locomotion channel");
                }
                require(
                    frame.pose.foot_contacts == locomotion.foot_contacts,
                    "active-arm return delayed locomotion foot contacts");
            }
            previous = frame;
            if (!frame.overrides_locomotion_pose) {
                require(
                    flat_pose_bits_equal(frame.pose, locomotion),
                    "active-arm return relinquished away from locomotion");
                relinquished = true;
                release_tick = tick;
                break;
            }
        }
        require(
            relinquished && release_tick >= 12 && release_tick <= 30,
            "post-PlaceRelease active-arm return did not converge after its bounded 0.50 second blend");
    }
}

void test_same_generation_free_lifecycle_ends_constraint_without_resurrection() {
    const FlatControllerPose locomotion = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        locomotion, make_pose(0.78125F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::PlaceReplay;
    output.diagnostics.target = {85U, 12U};
    output.diagnostics.affordance_id = 15U;
    output.diagnostics.hand = Hand::Right;
    output.diagnostics.object_state = ObjectState::Held;
    output.diagnostics.attached = true;
    output.diagnostics.hand_constraint_weight = 0.0F;
    const ControllerInteractionHandConstraint constraint =
        make_hand_constraint(output, locomotion, Hand::Right);

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        constraint);
    output.diagnostics.hand_constraint_weight = 1.0F;
    const ControllerInteractionFrameState held = handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        constraint);
    require(
        held.hand_constraint_result.applied,
        "same-generation lifecycle setup did not solve while Held");

    output.diagnostics.state = RuntimeState::PlaceRelease;
    output.diagnostics.object_state = ObjectState::Free;
    output.diagnostics.attached = false;
    const ControllerInteractionFrameState released = handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        constraint);
    require(
        !released.hand_constraint_validated &&
            !released.hand_constraint_result.applied,
        "same-generation Free lifecycle retained a semantic solve");

    output.diagnostics.state = RuntimeState::PlaceReplay;
    output.diagnostics.object_state = ObjectState::Held;
    output.diagnostics.attached = true;
    const ControllerInteractionFrameState stale_held = handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        constraint);
    require(
        !stale_held.hand_constraint_validated &&
            !stale_held.hand_constraint_result.applied,
        "same-generation Held rollback resurrected an ended constraint epoch");
}

void test_left_place_release_reset_clears_blend_and_rejects_free_entry_solve() {
    const FlatControllerPose locomotion = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        locomotion, make_pose(0.8125F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::PlaceReplay;
    output.diagnostics.target = {83U, 10U};
    output.diagnostics.affordance_id = 13U;
    output.diagnostics.hand = Hand::Left;
    output.diagnostics.object_state = ObjectState::Held;
    output.diagnostics.attached = true;
    output.diagnostics.hand_constraint_weight = 0.0F;
    const ControllerInteractionHandConstraint held_constraint =
        make_hand_constraint(output, locomotion, Hand::Left);

    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        held_constraint);
    output.diagnostics.hand_constraint_weight = 1.0F;
    const ControllerInteractionFrameState held = handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        held_constraint);
    require(
        held.hand_constraint_result.applied,
        "left held setup did not activate the semantic hand constraint");
    const FlatWorldPose held_world = flat_world_pose(held.pose);

    output.diagnostics.state = RuntimeState::PlaceRelease;
    output.diagnostics.target = {83U, 11U};
    output.diagnostics.object_state = ObjectState::Free;
    output.diagnostics.attached = false;
    output.diagnostics.hand_constraint_weight = 0.0F;
    const ControllerInteractionFrameState released = handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        std::nullopt);
    const FlatWorldPose released_world = flat_world_pose(released.pose);
    for (size_t bone = 15U; bone <= 18U; ++bone) {
        require_vec_near(
            released_world.positions[bone],
            held_world.positions[bone],
            "left PlaceRelease did not preserve active-arm world position");
        require_vec_near(
            released_world.velocities[bone],
            held_world.velocities[bone],
            "left PlaceRelease did not preserve active-arm world velocity");
        require_same_rotation(
            released_world.rotations[bone],
            held_world.rotations[bone],
            "left PlaceRelease did not preserve active-arm world rotation");
        require_vec_near(
            released_world.angular_velocities[bone],
            held_world.angular_velocities[bone],
            "left PlaceRelease did not preserve active-arm angular velocity");
    }

    FlatControllerPose authored_retract = locomotion;
    authored_retract.rotations[16] = quat_normalize(quat_mul(
        quat_from_angle_axis(
            0.10F, normalize(vec3(0.5F, 0.3F, 0.7F))),
        locomotion.rotations[16]));
    output.pose = interaction::expand_flat_controller_pose(
        authored_retract, raw_reference);
    const FlatControllerPose expected_authored_retract =
        interaction::collapse_interaction_pose(
            output.pose, raw_reference, locomotion);
    const ControllerInteractionFrameState retract = handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds,
        std::nullopt);
    bool left_arm_is_fully_authored = true;
    for (size_t bone = 15U; bone <= 18U; ++bone) {
        left_arm_is_fully_authored = left_arm_is_fully_authored &&
            flat_bone_channels_bits_equal(
                retract.pose, expected_authored_retract, bone);
    }
    require(
        !left_arm_is_fully_authored,
        "left active-arm release blend ended after one frame");

    handoff.reset();
    RuntimeOutput free_entry = make_owned_output(raw_reference);
    free_entry.diagnostics.state = RuntimeState::PlaceRelease;
    free_entry.diagnostics.target = {84U, 1U};
    free_entry.diagnostics.affordance_id = 14U;
    free_entry.diagnostics.hand = Hand::Left;
    free_entry.diagnostics.object_state = ObjectState::Free;
    free_entry.diagnostics.attached = false;
    free_entry.diagnostics.hand_constraint_weight = 1.0F;
    const ControllerInteractionHandConstraint free_constraint =
        make_hand_constraint(free_entry, locomotion, Hand::Left);
    const ControllerInteractionFrameState after_reset = handoff.apply(
        locomotion,
        free_entry,
        interaction::kControllerStepSeconds,
        free_constraint);
    require(
        !after_reset.hand_constraint_validated &&
            !after_reset.hand_constraint_result.applied,
        "reset re-entry started a semantic constraint on a Free object");
    require(
        flat_pose_bits_equal(after_reset.pose, locomotion),
        "reset re-entry retained a stale left-arm release overlay");
}

void test_place_release_clears_stale_inactive_arm_return_before_composite() {
    const FlatControllerPose entry = make_flat_pose();
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        entry, make_pose(0.84375F));
    RuntimeOutput output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.target = {86U, 2U};
    output.diagnostics.affordance_id = 16U;
    output.diagnostics.hand = Hand::Left;
    output.diagnostics.object_state = ObjectState::Held;
    output.diagnostics.attached = true;
    output.diagnostics.recorded_carry = false;
    output.diagnostics.inactive_arm_targets_locomotion = true;
    output.diagnostics.inactive_arm_tracks_locomotion = true;
    output.diagnostics.hand_constraint_weight = 0.0F;
    const ControllerInteractionHandConstraint constraint =
        make_hand_constraint(output, entry, Hand::Left);

    FlatControllerPose carry_locomotion = entry;
    for (size_t bone = 19U; bone <= 22U; ++bone) {
        carry_locomotion.rotations[bone] = quat_normalize(quat_mul(
            quat_from_angle_axis(
                0.20F + 0.02F * static_cast<float>(bone - 19U),
                normalize(vec3(0.3F, 0.8F, 0.4F))),
            entry.rotations[bone]));
    }
    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        carry_locomotion,
        output,
        interaction::kControllerStepSeconds,
        constraint);
    output.diagnostics.hand_constraint_weight = 1.0F;
    (void)handoff.apply(
        carry_locomotion,
        output,
        interaction::kControllerStepSeconds,
        constraint);

    FlatControllerPose authored_place = entry;
    for (size_t bone = 19U; bone <= 22U; ++bone) {
        authored_place.rotations[bone] = quat_normalize(quat_mul(
            quat_from_angle_axis(
                -0.18F - 0.015F * static_cast<float>(bone - 19U),
                normalize(vec3(0.7F, 0.2F, 0.5F))),
            entry.rotations[bone]));
    }
    output.pose = interaction::expand_flat_controller_pose(
        authored_place, raw_reference);
    output.diagnostics.recorded_carry = false;
    output.diagnostics.inactive_arm_targets_locomotion = false;
    output.diagnostics.inactive_arm_tracks_locomotion = false;
    for (RuntimeState state : {
             RuntimeState::PlacePreflight,
             RuntimeState::PlaceAlign,
             RuntimeState::PlaceReplay}) {
        output.diagnostics.state = state;
        (void)handoff.apply(
            entry,
            output,
            interaction::kControllerStepSeconds,
            constraint);
    }

    output.diagnostics.state = RuntimeState::PlaceRelease;
    output.diagnostics.target = {86U, 3U};
    output.diagnostics.object_state = ObjectState::Free;
    output.diagnostics.attached = false;
    output.diagnostics.hand_constraint_weight = 0.0F;
    const FlatControllerPose expected_release =
        interaction::collapse_interaction_pose(
            output.pose, raw_reference, carry_locomotion);
    const ControllerInteractionFrameState released = handoff.apply(
        entry,
        output,
        interaction::kControllerStepSeconds,
        std::nullopt);
    for (size_t bone = 19U; bone <= 22U; ++bone) {
        require(
            flat_bone_channels_bits_equal(
                released.pose, expected_release, bone),
            "PlaceRelease retained a stale inactive-arm return overlay");
    }
}

void test_frame_handoff_reset_and_reentry_capture_fresh_flat_reference() {
    const FlatControllerPose first_entry = make_flat_pose();
    RuntimeOutput owned = make_owned_output(make_pose(0.5F));
    ControllerInteractionFrameHandoff handoff;
    (void)handoff.apply(
        first_entry, owned, interaction::kControllerStepSeconds);

    handoff.reset();
    FlatControllerPose second_entry = first_entry;
    for (size_t bone = 0; bone < second_entry.positions.size(); ++bone) {
        const float value = static_cast<float>(bone + 1U);
        second_entry.positions[bone] = second_entry.positions[bone] +
            vec3(0.01F * value, 0.02F * value, -0.03F * value);
        second_entry.velocities[bone] = second_entry.velocities[bone] +
            vec3(-0.04F * value, 0.05F * value, 0.06F * value);
        second_entry.rotations[bone] = quat_from_angle_axis(
            0.01F * value, normalize(vec3(0.5F, 0.2F, 0.7F)));
        second_entry.angular_velocities[bone] =
            second_entry.angular_velocities[bone] +
            vec3(0.07F * value, -0.08F * value, 0.09F * value);
    }
    second_entry.foot_contacts = {0U, 1U};
    owned = make_owned_output(make_pose(1.5F));
    const ControllerInteractionFrameState after_reset = handoff.apply(
        second_entry, owned, interaction::kControllerStepSeconds);
    require(
        flat_pose_bits_equal(after_reset.pose, second_entry),
        "reset reused a stale ownership reference");

    RuntimeOutput idle;
    ControllerInteractionFrameState released{};
    for (int tick = 0; tick < 30; ++tick) {
        released = handoff.apply(
            second_entry, idle, interaction::kControllerStepSeconds);
        if (!released.overrides_locomotion_pose) {
            break;
        }
    }
    require(
        !released.overrides_locomotion_pose,
        "re-entry setup did not finish release");

    FlatControllerPose third_entry = second_entry;
    for (size_t bone = 0; bone < third_entry.positions.size(); ++bone) {
        const float value = static_cast<float>(bone + 1U);
        third_entry.positions[bone] = third_entry.positions[bone] +
            vec3(-0.03F * value, 0.01F * value, 0.02F * value);
        third_entry.velocities[bone] = third_entry.velocities[bone] +
            vec3(0.02F * value, -0.01F * value, 0.04F * value);
        third_entry.rotations[bone] = quat_from_angle_axis(
            0.02F * value, normalize(vec3(0.3F, 0.8F, 0.4F)));
        third_entry.angular_velocities[bone] =
            third_entry.angular_velocities[bone] +
            vec3(-0.05F * value, 0.03F * value, 0.02F * value);
    }
    third_entry.foot_contacts = {1U, 1U};
    const Pose third_raw_reference = make_pose(2.5F);
    owned = make_owned_output(third_raw_reference);
    const ControllerInteractionFrameState reentered = handoff.apply(
        third_entry, owned, interaction::kControllerStepSeconds);
    require(
        flat_pose_bits_equal(reentered.pose, third_entry),
        "re-entry reused a stale ownership flat reference");

    Pose third_raw_current = third_raw_reference;
    third_raw_current.positions[0] =
        third_raw_current.positions[0] + vec3(0.4F, 0.1F, -0.2F);
    third_raw_current.rotations[0] = quat_mul(
        quat_from_angle_axis(0.2F, vec3(0.0F, 1.0F, 0.0F)),
        third_raw_current.rotations[0]);
    owned.pose = third_raw_current;
    const ControllerInteractionFrameState advanced = handoff.apply(
        third_entry, owned, interaction::kControllerStepSeconds);
    const FlatControllerPose expected = interaction::collapse_interaction_pose(
        third_raw_current, third_raw_reference, third_entry);
    require_flat_pose_near(
        advanced.pose,
        expected,
        "re-entry reused a stale raw interaction reference");
}

void test_frame_handoff_preserves_fresh_complete_nonowned_pose_for_25_frames() {
    ControllerInteractionFrameHandoff disabled_handoff;
    ControllerInteractionFrameHandoff loaded_handoff;
    ControllerInteractionScheduler disabled_scheduler;
    ControllerInteractionScheduler loaded_scheduler;
    interaction::InteractionRuntime disabled_runtime =
        interaction::InteractionRuntime::disabled(
            Reason::PackUnavailable);
    interaction::RuntimeFixture loaded_fixture =
        interaction::make_runtime_fixture();
    interaction::InteractionRuntime loaded_runtime(
        loaded_fixture.database,
        loaded_fixture.features,
        loaded_fixture.registry,
        interaction::RuntimeConfig{});
    int disabled_runtime_calls = 0;
    int loaded_runtime_calls = 0;
    FlatControllerPose previous{};

    for (int tick = 1; tick <= 25; ++tick) {
        FlatControllerPose fresh = make_flat_pose();
        fresh.positions[0].x += static_cast<float>(tick);
        fresh.velocities[0].z -= static_cast<float>(tick);
        if (tick != 1) {
            assert(!flat_pose_bits_equal(fresh, previous));
        }
        previous = fresh;

        const RuntimeOutput& disabled_output = disabled_scheduler.tick(
            {},
            [&] { return make_snapshot(static_cast<float>(tick)); },
            [](const LocomotionSnapshot&) -> std::optional<PickRequest> {
                return std::nullopt;
            },
            [&](const RuntimeInput& input) {
                ++disabled_runtime_calls;
                assert(input.dt == interaction::kInteractionRuntimeStepSeconds);
                return disabled_runtime.update(input);
            });
        const RuntimeOutput& loaded_output = loaded_scheduler.tick(
            {},
            [&] { return make_snapshot(static_cast<float>(tick)); },
            [](const LocomotionSnapshot&) -> std::optional<PickRequest> {
                return std::nullopt;
            },
            [&](const RuntimeInput& input) {
                ++loaded_runtime_calls;
                assert(input.dt == interaction::kInteractionRuntimeStepSeconds);
                const RuntimeOutput output = loaded_runtime.update(input);
                assert(output.diagnostics.state == RuntimeState::Locomotion);
                assert(output.diagnostics.pack_available);
                assert(!output.owns_pose);
                return output;
            });
        const ControllerInteractionFrameState disabled_frame =
            disabled_handoff.apply(
                fresh,
                disabled_output,
                interaction::kControllerStepSeconds);
        const ControllerInteractionFrameState loaded_frame =
            loaded_handoff.apply(
                fresh,
                loaded_output,
                interaction::kControllerStepSeconds);
        assert(flat_pose_bits_equal(disabled_frame.pose, fresh));
        assert(flat_pose_bits_equal(loaded_frame.pose, fresh));
        assert(!disabled_frame.runtime_owns_pose);
        assert(!loaded_frame.runtime_owns_pose);
        assert(!disabled_frame.overrides_locomotion_pose);
        assert(!loaded_frame.overrides_locomotion_pose);
        assert(!disabled_frame.synchronize_simulation_root);
        assert(!loaded_frame.synchronize_simulation_root);
    }

    assert(disabled_runtime_calls == 25);
    assert(loaded_runtime_calls == 25);
    assert(disabled_scheduler.phase() == 0);
    assert(loaded_scheduler.phase() == 0);
}

void test_frame_handoff_exposes_rendered_flat_root_sync() {
    for (float value : interaction::kFlatControllerRestHandDof) {
        assert(float_bits_equal(value, 0.0F));
    }
    for (float value : interaction::kFlatControllerRestHandDofVelocities) {
        assert(float_bits_equal(value, 0.0F));
    }

    ControllerInteractionFrameHandoff handoff;
    const FlatControllerPose locomotion = make_flat_pose();
    FlatControllerPose target = locomotion;
    target.positions[0] = target.positions[0] + vec3(2.0F, 1.0F, -3.0F);
    target.rotations[6] = quat_from_angle_axis(
        0.55F, normalize(vec3(0.2F, 0.9F, 0.3F)));
    target.foot_contacts = {0U, 1U};
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        locomotion, make_pose(0.375F));
    RuntimeOutput owned = make_owned_output(raw_reference);
    (void)handoff.apply(
        locomotion, owned, interaction::kControllerStepSeconds);
    owned.pose = interaction::expand_flat_controller_pose(
        target, raw_reference);

    ControllerInteractionFrameState frame{};
    for (int tick = 1; tick <= 15; ++tick) {
        frame = handoff.apply(
            locomotion, owned, interaction::kControllerStepSeconds);
        require(frame.runtime_owns_pose, "owned frame lost runtime ownership");
        require(
            frame.overrides_locomotion_pose,
            "owned frame lost visual override");
        require(
            frame.synchronize_simulation_root,
            "pre-Carry frame did not request root synchronization");
        require(
            vec_bits_equal(
                frame.simulation_root_position, frame.pose.positions[0]),
            "published simulation root position differs from rendered flat root");
        require(
            quat_bits_equal(
                frame.simulation_root_rotation, frame.pose.rotations[0]),
            "published simulation root rotation differs from rendered flat root");
    }
    assert_flat_pose_near(frame.pose, target);
    assert(frame.pose.foot_contacts == target.foot_contacts);
    require_same_rotation(
        frame.pose.rotations[6],
        target.rotations[6],
        "pre-Carry ownership failed to publish the retargeted leg");

    vec3 simulation_position(91.0F, 92.0F, 93.0F);
    quat simulation_rotation = quat_from_angle_axis(
        0.3F, vec3(0.0F, 1.0F, 0.0F));
    if (frame.synchronize_simulation_root) {
        simulation_position = frame.simulation_root_position;
        simulation_rotation = frame.simulation_root_rotation;
    }
    assert(vec_bits_equal(simulation_position, frame.pose.positions[0]));
    assert(quat_bits_equal(simulation_rotation, frame.pose.rotations[0]));

    FlatControllerPose fresh_nonowned = make_flat_pose();
    fresh_nonowned.positions[0].x += 31.0F;
    RuntimeOutput nonowned;
    const ControllerInteractionFrameState released = handoff.apply(
        fresh_nonowned, nonowned, interaction::kControllerStepSeconds);
    require(
        released.overrides_locomotion_pose,
        "release dropped visual override before blending");
    assert_flat_pose_near(released.pose, frame.pose);
    require(
        !released.synchronize_simulation_root,
        "release incorrectly synchronized the simulation root");
}

void test_frame_handoff_keeps_layered_carry_simulation_root_live() {
    FlatControllerPose locomotion = make_flat_pose();
    FlatControllerPose target = locomotion;
    target.positions[0] = target.positions[0] + vec3(4.0F, 0.5F, 2.0F);
    target.rotations[6] = quat_from_angle_axis(
        0.72F, normalize(vec3(0.1F, 0.8F, 0.4F)));
    target.rotations[12] = quat_from_angle_axis(
        0.48F, normalize(vec3(0.2F, 0.6F, 0.7F)));
    const Pose raw_reference = interaction::expand_flat_controller_pose(
        locomotion, make_pose(0.375F));
    const Pose raw_target = interaction::expand_flat_controller_pose(
        target, raw_reference);
    RuntimeOutput output = make_owned_output(raw_reference);

    ControllerInteractionFrameHandoff layered_handoff;
    (void)layered_handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    FlatControllerPose hold_target = locomotion;
    hold_target.rotations[12] = target.rotations[12];
    output.pose = interaction::expand_flat_controller_pose(
        hold_target, raw_reference);
    output.diagnostics.state = RuntimeState::Hold;
    const FlatControllerPose pre_carry_rendered = layered_handoff.apply(
        locomotion,
        output,
        interaction::kControllerStepSeconds).pose;
    locomotion.positions[0] =
        locomotion.positions[0] + vec3(-1.0F, 0.0F, 0.5F);
    locomotion.rotations[6] = quat_from_angle_axis(
        0.11F, vec3(0.0F, 1.0F, 0.0F));
    output.pose = raw_target;
    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = false;
    ControllerInteractionFrameState layered = layered_handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    require(
        layered.runtime_owns_pose,
        "layered Carry lost runtime pose ownership");
    require(
        layered.overrides_locomotion_pose,
        "layered Carry lost visual override");
    require(
        !layered.synchronize_simulation_root,
        "layered Carry overwrote the live simulation root");
    require(
        vec_bits_equal(
            layered.pose.positions[0], pre_carry_rendered.positions[0]) &&
            quat_bits_equal(
                layered.pose.rotations[6], pre_carry_rendered.rotations[6]),
        "first layered Carry frame did not preserve the rendered root/leg base");

    bool layered_lower_converged = false;
    for (int tick = 0; tick < 64 && !layered_lower_converged; ++tick) {
        layered = layered_handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
        require(
            !layered.synchronize_simulation_root,
            "layered Carry synchronized the simulation root during handoff");
        layered_lower_converged =
            vec_bits_equal(
                layered.pose.positions[0], locomotion.positions[0]) &&
            quat_bits_equal(
                layered.pose.rotations[6], locomotion.rotations[6]);
    }
    require(
        layered_lower_converged,
        "layered Carry did not converge to the live root/leg base");

    output = make_owned_output(raw_reference);
    ControllerInteractionFrameHandoff pre_carry_handoff;
    (void)pre_carry_handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    output.pose = raw_target;
    const ControllerInteractionFrameState pre_carry =
        pre_carry_handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
    require(
        pre_carry.synchronize_simulation_root,
        "pre-Carry ownership did not synchronize the simulation root");
    require(
        vec_bits_equal(
            pre_carry.simulation_root_position, pre_carry.pose.positions[0]),
        "pre-Carry synchronized the wrong root position");
    require(
        quat_bits_equal(
            pre_carry.simulation_root_rotation, pre_carry.pose.rotations[0]),
        "pre-Carry synchronized the wrong root rotation");
    require(
        !vec_bits_equal(
            pre_carry.pose.positions[0], locomotion.positions[0]) &&
            !quat_bits_equal(
                pre_carry.pose.rotations[6], locomotion.rotations[6]),
        "pre-Carry ownership failed to publish full-body root/legs");

    output = make_owned_output(raw_reference);
    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = true;
    ControllerInteractionFrameHandoff recorded_handoff;
    (void)recorded_handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    output.pose = raw_target;
    const ControllerInteractionFrameState recorded = recorded_handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    require(
        recorded.synchronize_simulation_root,
        "recorded Carry did not synchronize the simulation root");
    require(
        vec_bits_equal(
            recorded.simulation_root_position, recorded.pose.positions[0]),
        "recorded Carry synchronized the wrong root position");
    require(
        quat_bits_equal(
            recorded.simulation_root_rotation, recorded.pose.rotations[0]),
        "recorded Carry synchronized the wrong root rotation");
    require(
        !vec_bits_equal(
            recorded.pose.positions[0], locomotion.positions[0]) &&
            !quat_bits_equal(
                recorded.pose.rotations[6], locomotion.rotations[6]),
        "recorded Carry failed to publish full-body root/legs");
}

void test_scene_handoff_publishes_each_fresh_25_hz_sample_without_lag() {
    ControllerInteractionScheduler scheduler;
    ControllerInteractionSceneHandoff handoff;
    InteractionTarget target;
    target.handle = {40, 2};
    target.object_profile_id = 4001U;
    target.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    target.object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.04F)};
    target.state = ObjectState::Held;
    target.object_world = {
        vec3(-3.0F, 0.5F, 4.0F),
        quat_from_angle_axis(-0.3F, vec3(0.0F, 1.0F, 0.0F))};
    const Transform authored_fallback = target.object_world;
    const InteractionTarget target_before = target;

    int update_calls = 0;
    for (int tick = 1; tick <= 8; ++tick) {
        const RuntimeOutput& output = scheduler.tick(
            {},
            [] { return LocomotionSnapshot{}; },
            [](const LocomotionSnapshot&) -> std::optional<PickRequest> {
                return std::nullopt;
            },
            [&](const RuntimeInput&) {
                RuntimeOutput next;
                next.diagnostics.target = target.handle;
                next.diagnostics.object_state = ObjectState::Held;
                const float sample = static_cast<float>(update_calls++);
                next.object_world = {
                    vec3(0.24F * sample, 1.0F, -2.0F),
                    quat_from_angle_axis(
                        0.2F * sample, vec3(0.0F, 1.0F, 0.0F))};
                return next;
            });
        const RuntimeOutput output_before = output;
        const ControllerInteractionSceneState scene = handoff.apply(
            &target,
            output,
            authored_fallback,
            0.0F,
            scheduler.updated_last_tick());

        require(
            transform_bits_equal(target.object_world, target_before.object_world),
            "scene handoff mutated the registry transform");
        require(
            output_fields_equal(output, output_before),
            "scene handoff mutated the fresh runtime sample");

        require(scene.runtime_authority, "held runtime sample lost authority");
        const float current_sample =
            0.24F * static_cast<float>(update_calls - 1);
        require(
            near(scene.object_world.position.x, current_sample, 2.0e-5F),
            "scene handoff lagged behind the fresh runtime translation");
        require(
            near(scene.object_world.position.y, 1.0F) &&
                near(scene.object_world.position.z, -2.0F),
            "scene handoff changed constant translation channels");
        const quat expected_rotation =
            quat_from_angle_axis(
                0.2F * static_cast<float>(update_calls - 1),
                vec3(0.0F, 1.0F, 0.0F));
        require_same_rotation(
            scene.object_world.rotation,
            expected_rotation,
            "scene handoff lagged behind the fresh runtime rotation");
        require(scheduler.phase() == 0, "synchronous scheduler phase drifted");
        require(
            scheduler.updated_last_tick(),
            "synchronous scheduler did not publish a fresh sample");
    }
}

void test_scene_handoff_publishes_fresh_equal_plateau_samples_exactly() {
    ControllerInteractionScheduler scheduler;
    ControllerInteractionSceneHandoff handoff;
    InteractionTarget target;
    target.handle = {44, 6};
    target.object_profile_id = 4001U;
    target.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    target.object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.04F)};
    target.state = ObjectState::Held;
    target.object_world = {
        vec3(-5.0F, 0.5F, 4.0F),
        quat_from_angle_axis(-0.2F, vec3(0.0F, 1.0F, 0.0F))};
    const Transform authored_fallback = target.object_world;
    int samples = 0;
    const std::array<float, 3> authoritative_x = {0.0F, 12.0F, 12.0F};
    for (size_t tick = 0; tick < authoritative_x.size(); ++tick) {
        const RuntimeOutput& output = scheduler.tick(
            {},
            [] { return LocomotionSnapshot{}; },
            [](const LocomotionSnapshot&) -> std::optional<PickRequest> {
                return std::nullopt;
            },
            [&](const RuntimeInput&) {
                RuntimeOutput next;
                next.diagnostics.target = target.handle;
                next.diagnostics.object_state = ObjectState::Held;
                next.object_world = {
                    vec3(authoritative_x.at(static_cast<size_t>(samples++)),
                         1.0F,
                         2.0F),
                    quat_from_angle_axis(
                        0.4F, vec3(0.0F, 1.0F, 0.0F))};
                return next;
            });
        require(
            scheduler.phase() == 0,
            "synchronous scheduler phase changed in plateau regression");
        require(
            scheduler.updated_last_tick(),
            "synchronous scheduler did not expose a fresh plateau sample");
        const ControllerInteractionSceneState scene = handoff.apply(
            &target,
            output,
            authored_fallback,
            0.0F,
            scheduler.updated_last_tick());
        require(
            near(
                scene.object_world.position.x,
                authoritative_x[tick],
                2.0e-5F),
            "fresh equal plateau sample was not published exactly");
    }
    require(samples == 3, "plateau regression did not deliver three samples");
}

void test_scene_handoff_publishes_fresh_rotation_and_holds_only_explicit_cache() {
    ControllerInteractionSceneHandoff handoff;
    InteractionTarget target;
    target.handle = {41, 3};
    target.object_profile_id = 4001U;
    target.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    target.object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.04F)};
    target.state = ObjectState::Attached;
    target.object_world = Transform{};
    const Transform authored_fallback = target.object_world;

    RuntimeOutput first;
    first.diagnostics.target = target.handle;
    first.object_world = {
        vec3(0.0F, 1.0F, 2.0F),
        quat_from_angle_axis(
            170.0F * 3.14159265358979323846F / 180.0F,
            vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState initialized = handoff.apply(
        &target, first, authored_fallback, 0.75F, true);
    require(
        initialized.runtime_authority &&
            transform_bits_equal(initialized.object_world, first.object_world),
        "first scene authority did not initialize both endpoints exactly");

    RuntimeOutput second = first;
    second.object_world = {
        vec3(10.0F, 1.0F, 2.0F),
        -quat_from_angle_axis(
            190.0F * 3.14159265358979323846F / 180.0F,
            vec3(0.0F, 1.0F, 0.0F))};
    const RuntimeOutput second_before = second;
    const ControllerInteractionSceneState fresh = handoff.apply(
        &target, second, authored_fallback, 0.25F, true);
    require_vec_near(
        fresh.object_world.position,
        second.object_world.position,
        "scene handoff lagged a fresh translation sample");
    require_same_rotation(
        fresh.object_world.rotation,
        second.object_world.rotation,
        "scene handoff lagged a fresh rotation sample");

    RuntimeOutput changed_but_not_fresh = second;
    changed_but_not_fresh.object_world.position.x = 20.0F;
    changed_but_not_fresh.object_world.rotation = quat_from_angle_axis(
        -0.4F, vec3(0.0F, 1.0F, 0.0F));
    const ControllerInteractionSceneState cached = handoff.apply(
        &target,
        changed_but_not_fresh,
        authored_fallback,
        0.75F,
        false);
    require_vec_near(
        cached.object_world.position,
        second.object_world.position,
        "non-fresh output replaced the explicit cached sample");
    require_same_rotation(
        cached.object_world.rotation,
        second.object_world.rotation,
        "non-fresh output replaced the explicit cached rotation");
    require(
        output_fields_equal(second, second_before),
        "scene handoff canonicalized the runtime quaternion in place");
}

void test_scene_handoff_retains_post_failure_held_pose_until_registry_reclaims_authority() {
    ControllerInteractionSceneHandoff handoff;
    InteractionTarget target;
    target.handle = {41, 3};
    target.object_profile_id = 4001U;
    target.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    target.object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.04F)};
    target.state = ObjectState::Attached;
    target.object_world = {
        vec3(2.0F, 0.8F, 3.0F),
        quat_from_angle_axis(0.2F, vec3(0.0F, 1.0F, 0.0F))};
    const Transform authored_fallback = target.object_world;

    RuntimeOutput attached;
    attached.diagnostics.target = target.handle;
    attached.diagnostics.attached = true;
    attached.object_world = {
        vec3(2.5F, 1.2F, 3.4F),
        quat_from_angle_axis(0.7F, vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState attached_scene = handoff.apply(
        &target, attached, authored_fallback, 0.6F, true);
    assert(attached_scene.runtime_authority);
    assert(transform_bits_equal(
        attached_scene.object_world, attached.object_world));

    target.state = ObjectState::Held;
    RuntimeOutput post_failure = attached;
    post_failure.diagnostics.attached = false;
    post_failure.object_world = {
        vec3(2.8F, 1.4F, 3.7F),
        quat_from_angle_axis(0.9F, vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState held_scene = handoff.apply(
        &target, post_failure, authored_fallback, 0.25F, true);
    assert(held_scene.runtime_authority);
    require_vec_near(
        held_scene.object_world.position,
        post_failure.object_world.position,
        "post-failure Held output lagged the fresh authoritative transform");
    require_same_rotation(
        held_scene.object_world.rotation,
        post_failure.object_world.rotation,
        "post-failure Held rotation lagged the fresh authoritative transform");

    InteractionTarget stale_generation = target;
    stale_generation.handle = {41, 4};
    stale_generation.object_world = {
        vec3(-1.0F, 0.75F, 5.0F),
        quat_from_angle_axis(0.1F, vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState stale_scene = handoff.apply(
        &stale_generation, post_failure, authored_fallback, 0.9F, true);
    require(
        !stale_scene.runtime_authority &&
            transform_bits_equal(
                stale_scene.object_world, stale_generation.object_world),
        "stale runtime generation retained scene authority");

    RuntimeOutput replacement_output = post_failure;
    replacement_output.diagnostics.target = stale_generation.handle;
    replacement_output.object_world = {
        vec3(4.0F, 1.1F, 6.0F),
        quat_from_angle_axis(-0.8F, vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState replacement_initialized =
        handoff.apply(
            &stale_generation,
            replacement_output,
            authored_fallback,
            0.9F,
            true);
    require(
        replacement_initialized.runtime_authority &&
            transform_bits_equal(
                replacement_initialized.object_world,
                replacement_output.object_world),
        "new target generation blended from stale target history");

    InteractionTarget replacement = stale_generation;
    replacement.state = ObjectState::Free;
    replacement.object_world = {
        vec3(-3.0F, 0.9F, 4.0F),
        quat_from_angle_axis(-0.5F, vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState free_scene = handoff.apply(
        &replacement, replacement_output, authored_fallback, 0.4F, false);
    assert(!free_scene.runtime_authority);
    assert(transform_bits_equal(
        free_scene.object_world, replacement.object_world));

    replacement.state = ObjectState::Targeted;
    replacement.object_world.position.x += 0.25F;
    const ControllerInteractionSceneState targeted_scene = handoff.apply(
        &replacement, replacement_output, authored_fallback, 0.8F, false);
    assert(!targeted_scene.runtime_authority);
    assert(transform_bits_equal(
        targeted_scene.object_world, replacement.object_world));

    const ControllerInteractionSceneState missing_scene = handoff.apply(
        nullptr, replacement_output, authored_fallback, 0.2F, false);
    require(
        !missing_scene.runtime_authority &&
            transform_bits_equal(missing_scene.object_world, authored_fallback),
        "missing target did not reset to the exact authored fallback");

    replacement.state = ObjectState::Held;
    const ControllerInteractionSceneState after_reset = handoff.apply(
        &replacement, replacement_output, authored_fallback, 0.95F, true);
    require(
        after_reset.runtime_authority &&
            transform_bits_equal(
                after_reset.object_world, replacement_output.object_world),
        "registry authority reset retained old interpolation history");
}

void test_first_free_scene_sample_is_bit_exact_final_attached_runtime_pose() {
    ControllerInteractionSceneHandoff handoff;
    InteractionTarget attached_target;
    attached_target.handle = {91U, 10U};
    attached_target.state = ObjectState::Held;
    attached_target.object_profile_id = 5001U;
    attached_target.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    attached_target.object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.04F)};
    attached_target.object_world = {
        vec3(-4.0F, 0.75F, 2.0F),
        quat_from_angle_axis(-0.25F, vec3(0.0F, 1.0F, 0.0F))};
    const Transform authored_fallback = attached_target.object_world;

    RuntimeOutput attached_output;
    attached_output.diagnostics.target = attached_target.handle;
    attached_output.diagnostics.object_state = ObjectState::Held;
    attached_output.diagnostics.attached = true;
    attached_output.object_world = {
        vec3(1.125F, 0.8125F, 4.375F),
        -quat_from_angle_axis(0.625F, vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState final_attached = handoff.apply(
        &attached_target,
        attached_output,
        authored_fallback,
        0.0F,
        true);
    require(
        final_attached.runtime_authority &&
            transform_bits_equal(
                final_attached.object_world, attached_output.object_world),
        "final attached runtime pose was not published exactly");

    InteractionTarget free_target = attached_target;
    free_target.handle = {91U, 11U};
    free_target.state = ObjectState::Free;
    free_target.object_world = attached_output.object_world;
    free_target.table_world = {
        vec3(0.0F, 0.35F, 4.20F), quat()};
    free_target.table_size = vec3(1.0F, 0.70F, 1.0F);
    RuntimeOutput stale_attached_output = attached_output;
    const ControllerInteractionSceneState first_free = handoff.apply(
        &free_target,
        stale_attached_output,
        authored_fallback,
        0.0F,
        false);
    require(
        !first_free.runtime_authority &&
            transform_bits_equal(
                first_free.object_world, final_attached.object_world),
        "first Free registry sample differed bit-for-bit from final attachment");
    require(
        transform_bits_equal(first_free.object_world, free_target.object_world),
        "first Free scene sample did not come directly from the registry");
}

void test_scene_handoff_rejects_invalid_alpha_atomically() {
    ControllerInteractionSceneHandoff handoff;
    InteractionTarget target;
    target.handle = {55, 7};
    target.object_profile_id = 4001U;
    target.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    target.object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.04F)};
    target.state = ObjectState::Held;
    target.object_world = {
        vec3(-2.0F, 0.5F, 3.0F),
        quat_from_angle_axis(0.3F, vec3(0.0F, 1.0F, 0.0F))};
    const InteractionTarget target_before = target;
    const Transform authored_fallback = target.object_world;

    RuntimeOutput first;
    first.diagnostics.target = target.handle;
    first.object_world = {
        vec3(0.0F, 1.0F, 2.0F),
        quat_from_angle_axis(0.1F, vec3(0.0F, 1.0F, 0.0F))};
    (void)handoff.apply(&target, first, authored_fallback, 0.0F, true);
    RuntimeOutput second = first;
    second.object_world = {
        vec3(10.0F, 1.0F, 2.0F),
        quat_from_angle_axis(0.5F, vec3(0.0F, 1.0F, 0.0F))};
    (void)handoff.apply(&target, second, authored_fallback, 0.5F, true);

    RuntimeOutput rejected = second;
    rejected.object_world = {
        vec3(20.0F, 1.0F, 2.0F),
        quat_from_angle_axis(0.9F, vec3(0.0F, 1.0F, 0.0F))};
    const RuntimeOutput rejected_before = rejected;
    const std::array<float, 4> invalid_alphas = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -0.01F,
        1.0F};
    for (float alpha : invalid_alphas) {
        bool threw = false;
        try {
            (void)handoff.apply(
                &target, rejected, authored_fallback, alpha, true);
        } catch (const interaction::FormatError&) {
            threw = true;
        }
        require(threw, "scene handoff accepted an invalid render alpha");
        require(
            transform_bits_equal(target.object_world, target_before.object_world),
            "invalid alpha mutated the registry target");
        require(
            output_fields_equal(rejected, rejected_before),
            "invalid alpha mutated the runtime output");
    }

    const ControllerInteractionSceneState unchanged = handoff.apply(
        &target, second, authored_fallback, 0.75F, false);
    require_vec_near(
        unchanged.object_world.position,
        second.object_world.position,
        "invalid alpha partially committed a new cached transform");
    require_same_rotation(
        unchanged.object_world.rotation,
        second.object_world.rotation,
        "invalid alpha partially committed a new cached rotation");
}

void test_carry_label_is_only_specific_during_carry() {
    RuntimeOutput output;
    output.diagnostics.recorded_carry = true;
    output.diagnostics.state = RuntimeState::Hold;
    assert(std::string(interaction::controller_carry_mode_label(output)) ==
           "none");
    output.diagnostics.state = RuntimeState::Carry;
    assert(std::string(interaction::controller_carry_mode_label(output)) ==
           "recorded");
    output.diagnostics.recorded_carry = false;
    assert(std::string(interaction::controller_carry_mode_label(output)) ==
           "layered");
}

void test_place_debug_status_prefers_active_controller_preview_values() {
    RuntimeOutput output;
    output.diagnostics.place.mode = interaction::PlaceMotionMode::RecordedPlace;
    output.diagnostics.place.preview_available = false;
    output.diagnostics.place.preview.accepted = false;
    output.diagnostics.place.preview.ready = true;
    output.diagnostics.place.preview.reason = Reason::SurfaceChanged;
    output.diagnostics.place.preview.root_error_m = 0.10F;
    output.diagnostics.place.preview.yaw_error_radians = 0.10F;

    PlaceStagingPreview staged;
    staged.accepted = true;
    staged.ready = false;
    staged.reason = Reason::None;
    staged.candidate.mode = interaction::PlaceMotionMode::ReversedPickup;
    staged.root_error_m = 0.80F;
    staged.yaw_error_radians = 0.60F;

    const interaction::RuntimePlaceDiagnostics active =
        interaction::controller_place_debug_diagnostics(output, staged);
    require(
        active.mode == interaction::PlaceMotionMode::ReversedPickup &&
            active.preview_available &&
            active.preview.accepted &&
            !active.preview.ready &&
            active.preview.reason == Reason::None &&
            active.preview.root_error_m >
                interaction::kPlaceStagingMaximumRootErrorM &&
            active.preview.yaw_error_radians >
                interaction::kPlaceStagingMaximumYawErrorRadians,
        "active far ReversedPickup staging did not own debug status values");

    const interaction::RuntimePlaceDiagnostics fallback =
        interaction::controller_place_debug_diagnostics(output, std::nullopt);
    require(
        fallback.mode == interaction::PlaceMotionMode::RecordedPlace &&
            !fallback.preview_available &&
            !fallback.preview.accepted &&
            fallback.preview.ready &&
            fallback.preview.reason == Reason::SurfaceChanged &&
            near(fallback.preview.root_error_m, 0.10F) &&
            near(fallback.preview.yaw_error_radians, 0.10F),
        "debug status did not fall back to runtime placement diagnostics");
}

Transform transform_from_arrays(
    const std::vector<float>& positions,
    const std::vector<float>& rotations,
    size_t index) {
    return {
        vec3(
            positions[index * 3],
            positions[index * 3 + 1],
            positions[index * 3 + 2]),
        quat(
            rotations[index * 4],
            rotations[index * 4 + 1],
            rotations[index * 4 + 2],
            rotations[index * 4 + 3])};
}

void test_demo_target_preserves_object_in_table_transform() {
    Database database;
    database.clip_count = 1;
    database.frame_count = 4;
    database.bone_count = g1_skeleton::BoneCount;
    database.range_starts = {0};
    database.range_stops = {4};
    database.hand_dof_count = 14U;
    database.phases = {
        static_cast<uint8_t>(Phase::Approach),
        static_cast<uint8_t>(Phase::Reach),
        static_cast<uint8_t>(Phase::Contact),
        static_cast<uint8_t>(Phase::Lift)};
    database.active_hands = {static_cast<uint8_t>(Hand::Left)};

    const Transform source_table{
        vec3(2.0F, 0.8F, -1.0F),
        quat_normalize(quat_mul(
            quat_from_angle_axis(0.65F, vec3(0.0F, 1.0F, 0.0F)),
            quat_from_angle_axis(0.06F, vec3(1.0F, 0.0F, 0.0F))))};
    const Transform object_in_table{
        vec3(0.35F, 0.45F, -0.22F),
        quat_from_angle_axis(-0.4F, vec3(0.0F, 1.0F, 0.0F))};
    const Transform source_object = compose(source_table, object_in_table);

    database.table_positions = {
        source_table.position.x,
        source_table.position.y,
        source_table.position.z};
    database.table_rotations = {
        source_table.rotation.w,
        source_table.rotation.x,
        source_table.rotation.y,
        source_table.rotation.z};
    database.table_sizes = {1.4F, 0.1F, 0.8F};
    database.object_dimensions = {0.12F, 0.25F, 0.16F};
    database.grasp_positions_object = {0.01F, 0.04F, -0.02F};
    const quat grasp_rotation =
        quat_from_angle_axis(0.2F, vec3(1.0F, 0.0F, 0.0F));
    database.grasp_rotations_object = {
        grasp_rotation.w,
        grasp_rotation.x,
        grasp_rotation.y,
        grasp_rotation.z};
    database.approach_directions_object = {0.0F, 0.0F, 1.0F};
    database.object_positions.resize(12, 0.0F);
    database.object_rotations.resize(16, 0.0F);
    for (size_t frame = 0; frame < 4; ++frame) {
        const Transform object = frame == 1 || frame == 2
            ? source_object
            : Transform{vec3(20.0F + static_cast<float>(frame), 0, 0), quat()};
        database.object_positions[frame * 3] = object.position.x;
        database.object_positions[frame * 3 + 1] = object.position.y;
        database.object_positions[frame * 3 + 2] = object.position.z;
        database.object_rotations[frame * 4] = object.rotation.w;
        database.object_rotations[frame * 4 + 1] = object.rotation.x;
        database.object_rotations[frame * 4 + 2] = object.rotation.y;
        database.object_rotations[frame * 4 + 3] = object.rotation.z;
    }

    const size_t pose_samples =
        static_cast<size_t>(database.frame_count) * g1_skeleton::BoneCount;
    database.positions.assign(pose_samples * 3U, 0.0F);
    database.velocities.assign(pose_samples * 3U, 0.0F);
    database.rotations.assign(pose_samples * 4U, 0.0F);
    database.angular_velocities.assign(pose_samples * 3U, 0.0F);
    for (size_t sample = 0; sample < pose_samples; ++sample) {
        database.rotations[sample * 4U] = 1.0F;
    }
    database.hand_dof.assign(
        static_cast<size_t>(database.frame_count) * 14U, 0.0F);
    database.hand_dof_velocities.assign(
        static_cast<size_t>(database.frame_count) * 14U, 0.0F);
    database.foot_contacts.assign(
        static_cast<size_t>(database.frame_count) * 2U, 0U);
    const Transform expected_grasp{
        vec3(0.04F, 0.10F, -0.03F),
        quat_from_angle_axis(0.31F, normalize(vec3(0.7F, 0.2F, 0.5F)))};
    const Transform contact_hand = compose(source_object, expected_grasp);
    const size_t contact_hand_sample =
        (2U * g1_skeleton::BoneCount +
         static_cast<size_t>(g1_skeleton::LeftWrist));
    database.positions[contact_hand_sample * 3U] = contact_hand.position.x;
    database.positions[contact_hand_sample * 3U + 1U] =
        contact_hand.position.y;
    database.positions[contact_hand_sample * 3U + 2U] =
        contact_hand.position.z;
    database.rotations[contact_hand_sample * 4U] = contact_hand.rotation.w;
    database.rotations[contact_hand_sample * 4U + 1U] =
        contact_hand.rotation.x;
    database.rotations[contact_hand_sample * 4U + 2U] =
        contact_hand.rotation.y;
    database.rotations[contact_hand_sample * 4U + 3U] =
        contact_hand.rotation.z;
    const Transform rest_object = transform_from_arrays(
        database.object_positions, database.object_rotations, 1U);
    const Transform expected_grasp_from_contact = compose(
        inverse(rest_object), contact_hand);

    const InteractionTarget target =
        interaction::make_controller_demo_target(database);
    assert(target.handle.id != 0U);
    assert(target.handle.generation != 0U);
    assert(target.object_profile_id != 0U);
    assert_vec_near(target.object_bounds.center_object, vec3());
    assert_vec_near(
        target.object_bounds.half_extents_object,
        target.object_dimensions * 0.5F);
    assert(near(target.table_world.position.x, 0.0F));
    assert(near(target.table_world.position.y, source_table.position.y));
    assert(near(target.table_world.position.z, 3.0F));
    assert_same_rotation(target.table_world.rotation, source_table.rotation);
    const Transform relocated_local =
        compose(inverse(target.table_world), target.object_world);
    assert_vec_near(relocated_local.position, object_in_table.position);
    assert_same_rotation(relocated_local.rotation, object_in_table.rotation);
    assert(target.affordances.size() == 1);
    assert(target.affordances[0].hand == Hand::Left);
    assert_vec_near(
        target.affordances[0].hand_in_object.position,
        expected_grasp_from_contact.position);
    assert_same_rotation(
        target.affordances[0].hand_in_object.rotation,
        expected_grasp_from_contact.rotation);
    assert(target.state == ObjectState::Free);

    assert_vec_near(rest_object.position, source_object.position);

    const vec3 source_normal = quat_mul_vec3(
        source_table.rotation, vec3(0.0F, 1.0F, 0.0F));
    assert(
        std::fabs(source_normal.x) > 0.02F ||
        std::fabs(source_normal.z) > 0.02F);
    const Transform source_surface = compose(
        source_table,
        Transform{
            vec3(0.0F, 0.5F * database.table_sizes[1], 0.0F), quat()});
    const Transform raw_object_in_surface = compose(
        inverse(source_surface), source_object);
    float expected_lowest_corner_m =
        std::numeric_limits<float>::infinity();
    for (int sign_x : {-1, 1}) {
        for (int sign_y : {-1, 1}) {
            for (int sign_z : {-1, 1}) {
                const vec3 corner_object =
                    target.object_bounds.center_object + vec3(
                        static_cast<float>(sign_x) *
                            target.object_bounds.half_extents_object.x,
                        static_cast<float>(sign_y) *
                            target.object_bounds.half_extents_object.y,
                        static_cast<float>(sign_z) *
                            target.object_bounds.half_extents_object.z);
                const vec3 corner_surface =
                    raw_object_in_surface.position + quat_mul_vec3(
                        raw_object_in_surface.rotation, corner_object);
                expected_lowest_corner_m = std::min(
                    expected_lowest_corner_m, corner_surface.y);
            }
        }
    }
    const float expected_bounds_clearance_m =
        expected_lowest_corner_m < 0.0F
        ? -expected_lowest_corner_m
        : 0.0F;
    const vec3 destination_normal = source_normal;

    const interaction::PlacementSurface destination =
        interaction::make_controller_demo_destination_surface(
            database, target);
    assert(destination.handle.id != 0U);
    assert(destination.handle.generation != 0U);
    assert_vec_near(
        destination.support_volume_world.position,
        target.table_world.position + vec3(0.0F, 0.0F, 1.20F) -
            expected_bounds_clearance_m * destination_normal);
    assert_same_rotation(
        destination.support_volume_world.rotation,
        target.table_world.rotation);
    assert_vec_near(destination.support_volume_size, target.table_size);
    assert_vec_near(
        destination.surface_world.position,
        destination.support_volume_world.position +
            destination_normal * (0.5F * target.table_size.y));
    assert_same_rotation(
        destination.surface_world.rotation,
        destination.support_volume_world.rotation);
    assert(near(destination.half_extent_x_m, 0.5F * target.table_size.x));
    assert(near(destination.half_extent_z_m, 0.5F * target.table_size.z));
    assert(near(destination.overhead_clearance_m, 2.00F));
    assert(destination.affordances.size() == 1U);

    const interaction::PlaceAffordance& place =
        destination.affordances.front();
    const interaction::PlacementFit authored_fit =
        interaction::evaluate_placement_fit(
            destination, place, target.object_bounds);
    assert(authored_fit.accepted);
    Transform expected_object_in_surface = raw_object_in_surface;
    expected_object_in_surface.position.y += expected_bounds_clearance_m;
    assert_vec_near(
        place.object_in_surface.position,
        expected_object_in_surface.position);
    assert_same_rotation(
        place.object_in_surface.rotation,
        expected_object_in_surface.rotation);
    const vec3 source_top = source_table.position +
        source_normal * (0.5F * database.table_sizes[1]);
    const vec3 to_top = source_top - source_object.position;
    const float along_normal =
        to_top.x * source_normal.x +
        to_top.y * source_normal.y +
        to_top.z * source_normal.z;
    const vec3 projected_support =
        source_object.position + along_normal * source_normal;
    const vec3 expected_support_object = compose(
        inverse(source_object), Transform{projected_support, quat()}).position;
    assert_vec_near(place.support_point_object, expected_support_object);
    const Transform destination_object = interaction::placement_goal_world(
        destination, place.object_in_surface);
    const vec3 destination_support = compose(
        destination_object,
        Transform{place.support_point_object, quat()}).position;
    const vec3 expected_support_surface = compose(
        expected_object_in_surface,
        Transform{expected_support_object, quat()}).position;
    assert_vec_near(
        destination_support,
        compose(
            destination.surface_world,
            Transform{expected_support_surface, quat()}).position,
        2.0e-5F);
    assert_vec_near(
        place.approach_direction_surface,
        vec3(0.0F, 1.0F, 0.0F));
}

void test_demo_scene_rejects_every_malformed_consumed_shape_as_format_error() {
    const Database valid =
        interaction::runtime_fixture_detail::make_database();
    const auto expect_format_error = [](
        const Database& malformed,
        const std::string& label) {
        try {
            (void)interaction::make_controller_demo_target(malformed);
        } catch (const interaction::FormatError&) {
            return;
        } catch (const std::exception& error) {
            throw std::runtime_error(
                label + " leaked a non-FormatError exception: " + error.what());
        }
        throw std::runtime_error(label + " was accepted");
    };
    const auto verify_exact_channel = [
        &valid,
        &expect_format_error](const char* label, auto member) {
        Database truncated = valid;
        auto& truncated_values = truncated.*member;
        truncated_values.pop_back();
        expect_format_error(
            truncated, std::string(label) + " truncated shape");

        Database extended = valid;
        auto& extended_values = extended.*member;
        extended_values.push_back(extended_values.back());
        expect_format_error(
            extended, std::string(label) + " extra-tail shape");
    };

    verify_exact_channel("range_starts", &Database::range_starts);
    verify_exact_channel("range_stops", &Database::range_stops);
    verify_exact_channel("active_hands", &Database::active_hands);
    verify_exact_channel("phases", &Database::phases);
    verify_exact_channel("positions", &Database::positions);
    verify_exact_channel("velocities", &Database::velocities);
    verify_exact_channel("rotations", &Database::rotations);
    verify_exact_channel(
        "angular_velocities", &Database::angular_velocities);
    verify_exact_channel("hand_dof", &Database::hand_dof);
    verify_exact_channel(
        "hand_dof_velocities", &Database::hand_dof_velocities);
    verify_exact_channel("foot_contacts", &Database::foot_contacts);
    verify_exact_channel("object_positions", &Database::object_positions);
    verify_exact_channel("object_rotations", &Database::object_rotations);
    verify_exact_channel("table_positions", &Database::table_positions);
    verify_exact_channel("table_rotations", &Database::table_rotations);
    verify_exact_channel("table_sizes", &Database::table_sizes);
    verify_exact_channel(
        "object_dimensions", &Database::object_dimensions);
    verify_exact_channel(
        "approach_directions_object",
        &Database::approach_directions_object);

    Database incompatible_hand_dof = valid;
    incompatible_hand_dof.hand_dof_count = 13U;
    expect_format_error(
        incompatible_hand_dof, "incompatible hand_dof_count 13");
    incompatible_hand_dof = valid;
    incompatible_hand_dof.hand_dof_count = 15U;
    expect_format_error(
        incompatible_hand_dof, "incompatible hand_dof_count 15");
}

void test_demo_destination_rejects_unsupported_fit_categories() {
    const Database database =
        interaction::runtime_fixture_detail::make_database();
    const InteractionTarget valid_target =
        interaction::make_controller_demo_target(database);
    const auto expect_format_error = [
        &database](const InteractionTarget& target, const char* label) {
        try {
            (void)interaction::make_controller_demo_destination_surface(
                database, target);
        } catch (const interaction::FormatError&) {
            return;
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string(label) +
                " leaked a non-FormatError exception: " + error.what());
        }
        throw std::runtime_error(
            std::string(label) + " destination was accepted");
    };

    InteractionTarget footprint_failure = valid_target;
    footprint_failure.table_size.x = 0.05F;
    footprint_failure.table_size.z = 0.05F;
    expect_format_error(footprint_failure, "footprint-invalid");

    InteractionTarget overhead_failure = valid_target;
    overhead_failure.object_bounds.half_extents_object =
        vec3(0.01F, 3.00F, 0.01F);
    expect_format_error(overhead_failure, "overhead-invalid");
}

void test_cross_pack_frame_count_validation() {
    Database database;
    Features features;
    database.frame_count = 25;
    features.frame_count = 24;
    bool threw = false;
    try {
        interaction::validate_controller_interaction_pack(database, features);
    } catch (const interaction::FormatError& error) {
        threw = true;
        assert(std::string(error.what()).find("frame count") !=
               std::string::npos);
    }
    assert(threw);
    features.frame_count = database.frame_count;
    interaction::validate_controller_interaction_pack(database, features);
}

void test_debug_draw_uses_real_correction_geometry_and_complete_text() {
    const std::string debug = read_project_text("interaction_debug_draw.h");
    assert(debug.find("requested_root_correction_m * right") ==
           std::string::npos);
    assert(debug.find(
               "displayed_pose.positions[0] - locomotion_pose.positions[0]") !=
           std::string::npos);
    assert(debug.find("target=%llu:%u affordance=%u") != std::string::npos);
    assert(debug.find("groups=[%.3f %.3f %.3f %.3f %.3f]") !=
           std::string::npos);
    assert(debug.find("root req/app=%.3f/%.3f") != std::string::npos);
    assert(debug.find("yaw req/app=%.3f/%.3f") != std::string::npos);
    assert(debug.find("hand pos/orient=%.3f/%.3f") != std::string::npos);
    assert(debug.find("object=%s carry=%s") != std::string::npos);
    assert(debug.find("controller_carry_mode_label(output)") !=
           std::string::npos);
    assert(debug.find("runtime=25Hz controller=25Hz") != std::string::npos);
    assert(debug.find("destination_surface->support_volume_world") !=
           std::string::npos);
    assert(debug.find("target->object_bounds.center_object") !=
           std::string::npos);
    assert(debug.find("placement_goal_world(") != std::string::npos);
    assert(debug.find("approach_direction_surface") != std::string::npos);
    assert(debug.find("staging_root_world") != std::string::npos);
    assert(debug.find("root/yaw=%.3f/%.3f ready=%d") !=
           std::string::npos);
    assert(debug.find(
               "controller_place_debug_diagnostics(output, staged_preview)") !=
           std::string::npos);
    assert(debug.find("preview=%d accepted=%d ready=%d reason=%s") !=
           std::string::npos);
    assert(debug.find("actual fit=%d gap=%.3f low/high=%.3f/%.3f") !=
           std::string::npos);
    assert(debug.find(
               "Interaction: F smart pickup/place  WASD/X cancel auto  R reset") !=
           std::string::npos);
}

void test_controller_smart_pickup_two_phase_production_seam() {
    const std::string controller = read_project_text("controller.cpp");
    const std::string adapter =
        read_project_text("interaction_controller_adapter.cpp");

    require(
        controller.find(
            "#include \"interaction_smart_pickup_controller.h\"") !=
                std::string::npos,
        "controller does not include the shared Smart Pickup coordinator");
    require(
        occurrence_count(
            controller,
            "manual_smart_pickup_controller.pre_step(") == 1U &&
            occurrence_count(
                controller,
                "manual_smart_pickup_controller.post_step(") == 1U,
        "controller does not invoke exactly one Smart Pickup pre/post pair");
    require(
        occurrence_count(
            controller, "interaction_scheduler.tick(") == 1U,
        "controller has more than one scheduler tick/publication seam");
    require(
        occurrence_count(
            controller,
            "interaction::LocomotionSnapshot live_flat_snapshot") == 1U,
        "controller has more than one post-step live-flat snapshot");

    const size_t press_edges_begin = controller.find(
        "        // Press edges and runtime updates share the fixed 25 Hz controller");
    const size_t cached_runtime_state = controller.find(
        "        const interaction::RuntimeState cached_interaction_state =",
        press_edges_begin);
    require(
        press_edges_begin != std::string::npos &&
            cached_runtime_state != std::string::npos &&
            press_edges_begin < cached_runtime_state,
        "controller lost the bounded 25 Hz press-edge seam");
    const std::string press_edges = controller.substr(
        press_edges_begin, cached_runtime_state - press_edges_begin);
    const std::string compact_press_edges =
        without_ascii_whitespace(press_edges);
    require(
        occurrence_count(
            press_edges,
            "const bool manual_smart_pickup_override_pressed") == 1U &&
            compact_press_edges.find(
                "constboolmanual_smart_pickup_override_pressed="
                "IsKeyPressed(KEY_W)||IsKeyPressed(KEY_A)||"
                "IsKeyPressed(KEY_S)||IsKeyPressed(KEY_D);") !=
                std::string::npos,
        "manual Smart Pickup override is not one bounded WASD press edge");

    const size_t manual_input_begin = controller.find(
        "        // Manual pick-assist input begins.");
    const size_t manual_input_end = controller.find(
        "        // Manual pick-assist input ends.", manual_input_begin);
    const size_t manual_observation_begin = controller.find(
        "        // Manual pick-assist observation begins.");
    const size_t manual_observation_end = controller.find(
        "        // Manual pick-assist observation ends.",
        manual_observation_begin);
    require(
        manual_input_begin != std::string::npos &&
            manual_input_end != std::string::npos &&
            manual_observation_begin != std::string::npos &&
            manual_observation_end != std::string::npos,
        "controller lost the bounded manual Smart Pickup source seams");
    const std::string manual_input = controller.substr(
        manual_input_begin, manual_input_end - manual_input_begin);
    const std::string manual_observation = controller.substr(
        manual_observation_begin,
        manual_observation_end - manual_observation_begin);
    const std::string compact_manual_input =
        without_ascii_whitespace(manual_input);
    const std::string compact_manual_observation =
        without_ascii_whitespace(manual_observation);

    require(
        compact_manual_input.find(
            "manual_smart_pickup_pre_input.cancel_pressed="
            "interaction_edges.cancel_pressed||"
            "interaction_edges.reset_pressed;") != std::string::npos &&
            manual_input.find("interaction_edges.reset_pressed = false") ==
                std::string::npos,
        "R reset is not forwarded as coordinator cancel while remaining "
        "scheduler-owned");
    require(
        compact_manual_input.find(
            "manual_smart_pickup_pre_input.manual_override_pressed="
            "manual_smart_pickup_override_pressed;") !=
            std::string::npos,
        "bounded WASD edge is not forwarded to Smart Pickup pre-step");

    require(
        compact_manual_input.find(
            "constinteraction::InteractionTarget*"
            "manual_smart_pickup_target=interaction_registry.find("
            "interaction_scene_target_handle);") != std::string::npos,
        "manual F does not capture the exact baked scene target");
    require(
        manual_input.find("resolve_single_target(") == std::string::npos &&
            manual_input.find("1.45F") == std::string::npos &&
            manual_input.find("make_pick_reach_waypoint(") ==
                std::string::npos &&
            manual_input.find("make_pick_entry_slots(") ==
                std::string::npos,
        "manual F retains proximity gating or fixed-two slot synthesis");
    require(
        controller.find("interaction::ControllerPickAssist") ==
                std::string::npos &&
            controller.find("manual_pick_assist") == std::string::npos,
        "controller retains a second manual assist state machine");

    const std::array<std::string, 12> ordered_pre_tokens{{
        "const interaction::InteractionTarget* manual_smart_pickup_target",
        "interaction_registry.find(",
        "manual_smart_pickup_pre_input.runtime_state =",
        "manual_smart_pickup_pre_input.interact_pressed =",
        "manual_smart_pickup_pre_input.cancel_pressed =",
        "manual_smart_pickup_pre_input.manual_override_pressed =",
        "manual_smart_pickup_pre_input.selected_target =",
        "manual_smart_pickup_pre_input.selected_affordance_id =",
        "manual_smart_pickup_pre_input.left_stick =",
        "manual_smart_pickup_pre_input.right_stick =",
        "manual_smart_pickup_controller.pre_step(",
        "gamepadstick_left = manual_smart_pickup_pre_step.left_stick",
    }};
    std::array<size_t, 12> pre_positions{};
    for (size_t index = 0U; index < ordered_pre_tokens.size(); ++index) {
        pre_positions[index] = manual_input.find(ordered_pre_tokens[index]);
    }
    require(
        std::all_of(
            pre_positions.begin(), pre_positions.end(),
            [](size_t position) { return position != std::string::npos; }) &&
            std::is_sorted(pre_positions.begin(), pre_positions.end()),
        "manual pre-step data does not flow lookup/input/call/returned-stick");
    const size_t returned_left = pre_positions.back();
    const size_t returned_right = manual_input.find(
        "gamepadstick_right = manual_smart_pickup_pre_step.right_stick",
        returned_left);
    require(
        returned_right != std::string::npos && returned_left < returned_right &&
            occurrence_count(
                manual_input,
                "manual_smart_pickup_controller.pre_step(") == 1U &&
            compact_manual_input.find(
                "manual_smart_pickup_controller.pre_step("
                "manual_smart_pickup_pre_input)") != std::string::npos,
        "manual pre-step result is not applied exactly once and in order");
    require(
        compact_manual_input.find(
            "if(manual_smart_pickup_pre_step.cancel_consumed||"
            "manual_smart_pickup_pre_step.manual_override_consumed||"
            "manual_smart_pickup_new_attempt){"
            "manual_pick_stationary_diagnostics={};}") !=
            std::string::npos,
        "manual override does not reset attempt-local stationary diagnostics");

    const size_t pre_step = controller.find(
        "manual_smart_pickup_controller.pre_step(");
    const size_t velocity_update = controller.find(
        "vec3 desired_velocity_curr = desired_velocity_update(", pre_step);
    const size_t simulation_update = controller.find(
        "        simulation_positions_update(", velocity_update);
    const size_t live_flat_snapshot = controller.find(
        "interaction::LocomotionSnapshot live_flat_snapshot",
        simulation_update);
    const size_t post_step = controller.find(
        "manual_smart_pickup_controller.post_step(",
        live_flat_snapshot);
    const size_t scheduler_tick = controller.find(
        "interaction_scheduler.tick(", post_step);
    require(
        pre_step != std::string::npos &&
            velocity_update != std::string::npos &&
            simulation_update != std::string::npos &&
            live_flat_snapshot != std::string::npos &&
            post_step != std::string::npos &&
            scheduler_tick != std::string::npos &&
            pre_step < velocity_update &&
            velocity_update < simulation_update &&
            simulation_update < live_flat_snapshot &&
            live_flat_snapshot < post_step &&
            post_step < scheduler_tick,
        "production activation is not pre/ordinary-step/bridge/post/scheduler");

    const std::string ordinary_step = controller.substr(
        velocity_update, live_flat_snapshot - velocity_update);
    const size_t manual_stationary_guard_begin = ordinary_step.find(
        "const bool manual_pick_stationary_constraint_active =");
    const size_t manual_stationary_guard_end = ordinary_step.find(
        "const bool manual_pick_stationary_constraint_latched_this_tick =",
        manual_stationary_guard_begin);
    require(
        manual_stationary_guard_begin != std::string::npos &&
            manual_stationary_guard_end != std::string::npos &&
            manual_stationary_guard_begin < manual_stationary_guard_end,
        "ordinary step lost the bounded manual stationary guard");
    const std::string compact_manual_stationary_guard =
        without_ascii_whitespace(ordinary_step.substr(
            manual_stationary_guard_begin,
            manual_stationary_guard_end - manual_stationary_guard_begin));
    require(
        compact_manual_stationary_guard.find(
            "constboolmanual_pick_stationary_constraint_active="
            "!legacy_interaction_fixture_mode&&"
            "!manual_smart_pickup_pre_step.cancel_consumed&&"
            "!manual_smart_pickup_pre_step.manual_override_consumed&&"
            "!interaction_edges.reset_pressed&&"
            "manual_smart_pickup_post_step.assist_output."
            "stationary_constraint&&cached_interaction_state=="
            "interaction::RuntimeState::Locomotion;") !=
            std::string::npos,
        "WASD override can reuse the prior Smart Pickup stationary constraint");
    require(
        occurrence_count(
            ordinary_step, "simulation_positions_update(") == 1U &&
            occurrence_count(
                ordinary_step, "simulation_rotations_update(") == 1U &&
            ordinary_step.find("obstacles_positions,") !=
                std::string::npos &&
            ordinary_step.find("obstacles_scales);") !=
                std::string::npos,
        "activation does not reuse one ordinary collision-aware locomotion step");
    require(
        manual_input.find(
            "gamepadstick_left = manual_smart_pickup_pre_step.left_stick") !=
                std::string::npos &&
            manual_input.find(
                "gamepadstick_right = manual_smart_pickup_pre_step.right_stick") !=
                std::string::npos,
        "ordinary activation does not consume the coordinator's zeroed sticks");

    const std::array<std::string, 13> ordered_post_tokens{{
        "interaction::SmartPickupPostStepInput manual_smart_pickup_post_input",
        "manual_smart_pickup_post_input.runtime_state =",
        "manual_smart_pickup_post_input.live_flat_snapshot =",
        "const interaction::InteractionTarget* manual_smart_pickup_current_target",
        "interaction_registry.find(",
        "manual_smart_pickup_post_input.current_target =",
        "manual_smart_pickup_post_input.obstacle_centers.push_back(",
        "manual_smart_pickup_post_input.obstacle_sizes.push_back(",
        "manual_smart_pickup_post_input.simulation_velocity =",
        "manual_smart_pickup_post_input.displayed_planar_speed_mps =",
        "manual_smart_pickup_post_input.camera_azimuth =",
        "manual_smart_pickup_post_input.next_request_id =",
        "preview_manual_smart_pickup",
    }};
    std::array<size_t, 13> post_positions{};
    for (size_t index = 0U; index < ordered_post_tokens.size(); ++index) {
        post_positions[index] =
            manual_observation.find(ordered_post_tokens[index]);
    }
    require(
        std::all_of(
            post_positions.begin(), post_positions.end(),
            [](size_t position) { return position != std::string::npos; }) &&
            std::is_sorted(post_positions.begin(), post_positions.end()),
        "manual post-step fields do not flow snapshot/target/obstacles/metrics/id");
    require(
        compact_manual_observation.find(
            "for(intobstacle_index=0;"
            "obstacle_index<obstacles_positions.size;++obstacle_index)") !=
                std::string::npos &&
            compact_manual_observation.find(
                "manual_smart_pickup_post_input.obstacle_centers.push_back("
                "obstacles_positions(obstacle_index));") !=
                std::string::npos &&
            compact_manual_observation.find(
                "manual_smart_pickup_post_input.obstacle_sizes.push_back("
                "obstacles_scales(obstacle_index));") !=
                std::string::npos,
        "post input does not copy the locomotion arrays element-for-element");
    require(
        occurrence_count(
            manual_observation,
            "interaction_runtime.preview_pick(") == 1U &&
            compact_manual_observation.find(
                "returninteraction_runtime.preview_pick("
                "snapshot,prospective_root,target,affordance_id);") !=
                std::string::npos,
        "manual callback is not one direct four-argument runtime preview");

    const size_t post_call = manual_observation.find(
        "manual_smart_pickup_controller.post_step(",
        post_positions.back());
    const size_t request_guard = manual_observation.find(
        "if (manual_smart_pickup_post_step.pick_request.has_value())",
        post_call);
    const size_t request_edge = manual_observation.find(
        "interaction_edges.interact_pressed = true", request_guard);
    const size_t request_latch = manual_observation.find(
        "manual_smart_pickup_request = "
        "manual_smart_pickup_post_step.pick_request",
        request_edge);
    require(
        post_call != std::string::npos &&
            compact_manual_observation.find(
                "manual_smart_pickup_controller.post_step("
                "manual_smart_pickup_post_input,"
                "preview_manual_smart_pickup)") != std::string::npos &&
            request_guard != std::string::npos &&
            request_edge != std::string::npos &&
            request_latch != std::string::npos &&
            post_call < request_guard && request_guard < request_edge &&
            request_edge < request_latch &&
            manual_observation.find(
                "++interaction_next_request_id", request_latch) ==
                std::string::npos,
        "post-step must latch its request without advancing the request ID");
    require(
        compact_manual_observation.find(
            "manual_smart_pickup_post_input.next_request_id="
            "interaction_next_request_id;") != std::string::npos &&
            compact_manual_observation.find(
                "manual_smart_pickup_post_step.snapshot_fingerprint=="
                "live_flat_snapshot_fingerprint") != std::string::npos,
        "post-step does not share the authoritative request ID/fingerprint");

    const std::string pick_resolver_marker =
        "                [&](const interaction::LocomotionSnapshot& snapshot)\n"
        "                    -> std::optional<interaction::PickRequest>";
    const size_t pick_resolver = controller.find(
        pick_resolver_marker, scheduler_tick);
    require(
        pick_resolver != std::string::npos,
        "cannot bound the scheduler's production locomotion provider");
    const size_t place_resolver = controller.find(
        "                [&](const interaction::LocomotionSnapshot& snapshot)\n"
        "                    -> std::optional<interaction::ControllerPlaceTarget>",
        pick_resolver);
    require(
        place_resolver != std::string::npos,
        "cannot bound the scheduler's production pick resolver");
    const std::string pick_resolver_source = controller.substr(
        pick_resolver, place_resolver - pick_resolver);
    const std::string compact_pick_resolver =
        without_ascii_whitespace(pick_resolver_source);
    std::string manual_request_local;
    size_t exchange_request = std::string::npos;
    for (const std::string& candidate : {
             std::string("manual_pick_request"),
             std::string("manual_smart_pickup_submission")}) {
        const std::string request_exchange =
            "conststd::optional<interaction::PickRequest>" + candidate +
            "=std::exchange(manual_smart_pickup_request,std::nullopt);";
        exchange_request = compact_pick_resolver.find(request_exchange);
        if (exchange_request != std::string::npos) {
            manual_request_local = candidate;
            break;
        }
    }
    const size_t validate_request = compact_pick_resolver.find(
        manual_request_local +
            "->request_id!=interaction_next_request_id",
        exchange_request);
    const size_t increment_request = compact_pick_resolver.find(
        "++interaction_next_request_id;", validate_request);
    const size_t return_request = compact_pick_resolver.find(
        "return" + manual_request_local + ";", increment_request);
    require(
        exchange_request != std::string::npos &&
            validate_request != std::string::npos &&
            increment_request != std::string::npos &&
            return_request != std::string::npos &&
            occurrence_count(
                compact_pick_resolver,
                "std::exchange(manual_smart_pickup_request,std::nullopt)") ==
                1U &&
            occurrence_count(
                compact_pick_resolver,
                "++interaction_next_request_id;") == 1U &&
            exchange_request < validate_request &&
            validate_request < increment_request &&
            increment_request < return_request,
        "manual resolver must exchange once, validate the pre-increment ID, "
        "increment once on consumption, and return the local request");
    const std::string provider_source = controller.substr(
        scheduler_tick, pick_resolver - scheduler_tick);
    const std::string compact_provider =
        without_ascii_whitespace(provider_source);
    const size_t provider_alias = compact_provider.find(
        "constinteraction::LocomotionSnapshot&snapshot="
        "live_flat_snapshot;");
    const size_t provider_return = compact_provider.find(
        "returnsnapshot;", provider_alias);
    const size_t scheduler_updated = controller.find(
        "interaction_scheduler.updated_last_tick()", scheduler_tick);
    require(
        provider_alias != std::string::npos &&
            provider_return != std::string::npos &&
            occurrence_count(provider_source, "return snapshot;") == 1U &&
            provider_source.find(
                "manual_smart_pickup_controller.post_step(") ==
                std::string::npos &&
            compact_provider.find("live_flat_snapshot=") ==
                std::string::npos &&
            scheduler_updated != std::string::npos &&
            provider_alias < provider_return &&
            scheduler_tick < scheduler_updated,
        "scheduler does not publish the sole post-step live-flat snapshot");

    const size_t scheduler_cleanup_end = controller.find(
        "        const uint64_t placement_preview_call_delta =",
        scheduler_tick);
    const std::string scheduler_call_end_marker = "                });";
    const size_t scheduler_call_end = controller.rfind(
        scheduler_call_end_marker, scheduler_cleanup_end);
    require(
        scheduler_cleanup_end != std::string::npos &&
            scheduler_call_end != std::string::npos &&
            scheduler_tick < scheduler_call_end,
        "cannot bound immediate post-scheduler manual cleanup");
    const std::string scheduler_cleanup = controller.substr(
        scheduler_call_end + scheduler_call_end_marker.size(),
        scheduler_cleanup_end -
            (scheduler_call_end + scheduler_call_end_marker.size()));
    const size_t clear_unconsumed = scheduler_cleanup.find(
        "manual_smart_pickup_request.reset();");
    const size_t reset_guard = scheduler_cleanup.find(
        "if (interaction_edges.reset_pressed)", clear_unconsumed);
    const size_t clear_post_output = scheduler_cleanup.find(
        "manual_smart_pickup_post_step = {};", reset_guard);
    require(
        clear_unconsumed != std::string::npos &&
            reset_guard != std::string::npos &&
            clear_post_output != std::string::npos &&
            occurrence_count(
                scheduler_cleanup,
                "manual_smart_pickup_request.reset();") == 1U &&
            occurrence_count(
                scheduler_cleanup,
                "manual_smart_pickup_post_step = {};") == 1U &&
            clear_unconsumed < reset_guard &&
            reset_guard < clear_post_output,
        "scheduler tick does not discard an unconsumed manual latch and clear "
        "persistent post output under reset");

    const size_t manual_diagnostics_begin = controller.find(
        "        // Manual pick-assist diagnostics begins.");
    const size_t manual_diagnostics_end = controller.find(
        "        // Manual pick-assist diagnostics ends.",
        manual_diagnostics_begin);
    require(
        manual_diagnostics_begin != std::string::npos &&
            manual_diagnostics_end != std::string::npos,
        "controller lost the bounded manual Smart Pickup diagnostics draw");
    const std::string compact_manual_diagnostics = without_ascii_whitespace(
        controller.substr(
            manual_diagnostics_begin,
            manual_diagnostics_end - manual_diagnostics_begin));
    require(
        compact_manual_diagnostics.find(
            "if(manual_pick_diagnostics.state=="
            "interaction::PickAssistState::SlotApproach||"
            "manual_pick_diagnostics.state=="
            "interaction::PickAssistState::Settling||"
            "manual_pick_diagnostics.state=="
            "interaction::PickAssistState::FinalPreview||"
            "manual_pick_diagnostics.state=="
            "interaction::PickAssistState::ReadyToSubmit){"
            "DrawText(\"SMARTPICKUPAUTO-WASDorXcancels\","
            "340,242,18,ORANGE);}") != std::string::npos,
        "active Smart Pickup ownership copy is missing or not state-bounded");

    const size_t placement_scheduler = adapter.find(
        "const PlaceTargetResolver& place_target_resolver");
    const size_t scheduler_end = adapter.find(
        "int ControllerInteractionScheduler::phase() const",
        placement_scheduler);
    require(
        placement_scheduler != std::string::npos &&
            scheduler_end != std::string::npos,
        "cannot isolate the production placement scheduler overload");
    const std::string scheduler_source = adapter.substr(
        placement_scheduler, scheduler_end - placement_scheduler);
    require(
        occurrence_count(
            scheduler_source,
            "input.locomotion = locomotion_provider();") == 1U &&
            occurrence_count(
                scheduler_source,
                "RuntimeOutput next_output = runtime_update(input);") == 1U,
        "one scheduler tick does not call one provider and publish one update");
}

void test_controller_and_make_clock_policy() {
    const std::string controller = read_project_text("controller.cpp");
    assert(controller.find("SetTargetFPS(25);") != std::string::npos);
    assert(controller.find("SetTargetFPS(60);") == std::string::npos);
    assert(controller.find(
               "const float dt = interaction::kControllerStepSeconds;") !=
           std::string::npos);
    assert(controller.find("ControllerInteractionScheduler") !=
           std::string::npos);
    const size_t scheduler_tick =
        controller.find("interaction_scheduler.tick(");
    assert(scheduler_tick != std::string::npos);
    assert(controller.find("interaction_frame_handoff.apply(") !=
           std::string::npos);
    const size_t scene_handoff_apply =
        controller.find("interaction_scene_handoff.apply(");
    const size_t frame_handoff_apply =
        controller.find("interaction_frame_handoff.apply(");
    require(
        scene_handoff_apply < frame_handoff_apply,
        "controller resolves scene authority after pose handoff");
    require(
        controller.find("const float interaction_scene_alpha =") ==
            std::string::npos,
        "synchronous scene publication still computes render interpolation");
    const size_t scene_sample_updated = controller.find(
        "const bool interaction_scene_sample_updated =");
    require(
        scene_sample_updated != std::string::npos &&
            scheduler_tick < scene_sample_updated &&
            scene_sample_updated < scene_handoff_apply &&
            controller.find(
                "interaction_scheduler.updated_last_tick()",
                scene_sample_updated) < scene_handoff_apply,
        "controller does not capture the explicit post-tick sample signal");
    require(
        controller.find(
            "0.0F,\n"
            "                interaction_scene_sample_updated);",
            scene_handoff_apply) <
            frame_handoff_apply,
        "controller does not publish the fresh scene sample without lag");
    const size_t exact_affordance =
        controller.find("interaction_registry.find_affordance(");
    require(
        exact_affordance != std::string::npos &&
            controller.find(
                "interaction_output.diagnostics.target",
                exact_affordance) != std::string::npos &&
            controller.find(
                "interaction_output.diagnostics.affordance_id",
                exact_affordance) != std::string::npos,
        "controller does not resolve the exact selected affordance");
    const size_t semantic_constraint = controller.find(
        "interaction::ControllerInteractionHandConstraint");
    require(
        semantic_constraint != std::string::npos &&
            controller.find(
                "interaction_scene_state.object_world,\n"
                "                selected_affordance->hand_in_object",
                semantic_constraint) != std::string::npos,
        "controller does not compose the scene-authoritative semantic grasp");
    require(
        controller.find(
            "interaction_hand_constraint);",
            frame_handoff_apply) != std::string::npos,
        "controller does not pass the semantic constraint into pose handoff");
    const size_t scene_draw = controller.find(
        "interaction::debug_draw::draw_interaction_scene(");
    require(
        scene_draw != std::string::npos &&
            controller.find(
                "interaction_scene_state.object_world,", scene_draw) !=
                std::string::npos,
        "controller does not draw the same fresh scene transform");
    require(
        controller.find("interaction_frame_state.pose.positions[bone]") !=
            std::string::npos,
        "controller does not publish the final handoff pose");
    require(
        controller.find("latest_owned_interaction_pose") ==
            std::string::npos,
        "controller retains a dynamic owned-pose bridge reference");
    const size_t canonical_world_initialization = controller.find(
        "        initialize_autodemo_canonical_world();");
    const size_t fixed_flat_reference = controller.find(
        "const interaction::FlatControllerPose\n"
        "        interaction_flat_reference_pose = [&]()");
    require(
        canonical_world_initialization != std::string::npos &&
            fixed_flat_reference != std::string::npos &&
            canonical_world_initialization < fixed_flat_reference,
        "controller does not freeze the flat reference after canonical init");
    require(
        controller.find(
            "reference.velocities.fill(vec3());", fixed_flat_reference) !=
                std::string::npos &&
            controller.find(
                "reference.angular_velocities.fill(vec3());",
                fixed_flat_reference) != std::string::npos,
        "controller fixed flat reference retains dynamic channels");
    for (const std::string channel : {
             "positions", "velocities", "rotations",
             "angular_velocities"}) {
        require(
            controller.find(
                "interaction_reference_pose." + channel +
                    "[interaction_root] =\n"
                    "        interaction_flat_reference_pose." + channel +
                    "[0];",
                fixed_flat_reference) != std::string::npos,
            "controller does not scene-align every fixed G1 root channel");
    }
    require(
        controller.find(
            "flat_locomotion_pose,\n"
            "                interaction_reference_pose,\n"
            "                interaction_flat_reference_pose);") !=
                std::string::npos &&
            controller.find(
                "interaction_frame_state.pose,\n"
                "                interaction_reference_pose,\n"
                "                interaction_flat_reference_pose);") !=
                std::string::npos,
        "controller does not use its fixed reference pair for live and debug poses");
    require(
        controller.find(
            "if (use_autodemo_canonical_snapshot)\n"
            "                    {\n"
            "                        return autodemo_canonical_entry->snapshot;") !=
            std::string::npos,
        "controller lost the canonical pickup snapshot exception");
    require(
        controller.find("interaction_debug_pose") != std::string::npos,
        "controller does not isolate reconstructed G1 debug pose");
    assert(controller.find("interaction_scene_handoff.apply(") !=
           std::string::npos);
    assert(controller.find("interaction_registry.find_by_id(") !=
           std::string::npos);
    require(
        controller.find(
            "interaction_output.diagnostics.target.id ==\n"
            "                interaction_scene_target_handle.id") ==
            std::string::npos,
        "scene refresh rejects an exact selected target with a different ID");
    assert(controller.find("++next_handle.generation") == std::string::npos);
    assert(controller.find("interaction::kControllerStepSeconds") !=
           std::string::npos);
    assert(controller.find("GetFrameTime(") == std::string::npos);
    assert(controller.find("runtime_accumulator") == std::string::npos);
    assert(controller.find("runtime_output_alpha") == std::string::npos);
    assert(controller.find(
               "emscripten_set_main_loop_arg(update_callback, &u, 25, 1);") !=
           std::string::npos);
    assert(controller.find(
               "emscripten_set_main_loop_arg(update_callback, &u, 60, 1);") ==
           std::string::npos);
    assert(controller.find(
               "emscripten_set_main_loop_arg(update_callback, &u, 0, 1);") ==
           std::string::npos);
    assert(controller.find("InteractionRuntime::disabled(") !=
           std::string::npos);
    assert(controller.find("catch (const interaction::FormatError&") !=
           std::string::npos);
    require(
        controller.find("const bool lmm_enabled = false;") !=
                std::string::npos &&
            controller.find("&lmm_enabled") == std::string::npos,
        "the unretimed learned matcher can still be enabled at runtime");
    require(
        controller.find("if      (f < 50)") != std::string::npos &&
            controller.find("else if (f < 75)") != std::string::npos &&
            controller.find("else if (f < 100)") != std::string::npos &&
            controller.find("else if (f < 125)") != std::string::npos &&
            controller.find("g_frame >= 167") != std::string::npos,
        "MM_DISCRETE input phases do not preserve their 25 Hz durations");

    const std::string database = read_project_text("database.h");
    require(
        database.find("locomotion_timing::kTrajectoryFrameOffsets") !=
            std::string::npos,
        "database trajectory features do not share the 25 Hz timing contract");
    require(
        database.find("database_trajectory_index_clamp(db, i, 20)") ==
                std::string::npos &&
            database.find("database_trajectory_index_clamp(db, i, 40)") ==
                std::string::npos &&
            database.find("database_trajectory_index_clamp(db, i, 60)") ==
                std::string::npos,
        "database trajectory features retain 60 Hz frame offsets");
    require(
        controller.find("locomotion_timing::kTrajectorySampleTimesSeconds") !=
                std::string::npos &&
            controller.find("locomotion_timing::kTrajectoryStepSeconds") !=
                std::string::npos,
        "controller trajectory prediction does not use the shared horizons");
    assert(controller.find("20.0f * dt") == std::string::npos);

    const std::string makefile = read_project_text("Makefile");
    const size_t sources_begin = makefile.find("INTERACTION_SOURCES :=");
    const size_t sources_end = makefile.find("HEADER =", sources_begin);
    assert(sources_begin != std::string::npos);
    assert(sources_end != std::string::npos);
    const std::string controller_sources =
        makefile.substr(sources_begin, sources_end - sources_begin);
    assert(controller_sources.find("interaction_runtime.cpp") !=
           std::string::npos);
    assert(controller_sources.find("interaction_controller_adapter.cpp") !=
           std::string::npos);
    assert(controller_sources.find("interaction_pose.cpp") !=
           std::string::npos);
    assert(controller_sources.find("interaction_target.cpp") !=
           std::string::npos);
    assert(controller_sources.find("_probe.cpp") == std::string::npos);
    assert(controller_sources.find("SOURCE := controller.cpp") !=
           std::string::npos);
    assert(makefile.find("CONTROLLER_CXXFLAGS := -std=c++17") !=
           std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 &&
        std::strcmp(argv[1], "--post-release-arm-fast-math-canary") == 0) {
        test_place_release_active_arm_return_is_bounded_after_ownership();
        return 0;
    }
    test_exact_constants();
    test_flat_bridge_exact_parent_tree_anchor_map_and_unmapped_head();
    test_flat_bridge_expansion_matches_every_world_anchor_and_retains_reference();
    test_calibrated_flat_bridge_reference_is_exact_true_g1_pose();
    test_calibrated_flat_bridge_transfers_supported_world_deltas();
    test_calibrated_flat_bridge_round_trips_representable_flat_pose();
    test_calibrated_flat_bridge_projects_nonrepresentable_flat_morphology();
    test_calibrated_flat_bridge_is_antipodal_invariant();
    test_calibrated_flat_bridge_rejects_invalid_input_without_mutation();
    test_reference_retarget_raw_reference_is_exact_flat_reference();
    test_reference_retarget_fixes_nonroot_translations_under_mismatched_source();
    test_reference_retarget_transfers_mapped_world_rotation_delta();
    test_reference_retarget_keeps_neck_and_head_reference_locals();
    test_flat_bridge_accepts_antipodal_equivalent_rotations();
    test_flat_bridge_rejects_invalid_input_without_mutation();
    test_flat_bridge_canaries_prove_exact_23_element_bounds();
    test_scheduler_cadence_and_cache();
    test_edges_latch_coalesce_and_clear_after_delivery();
    test_manual_pick_request_reset_priority_discards_stale_latch();
    test_cache_changes_only_after_successful_due_delivery();
    test_scheduler_latches_carry_place_and_submits_only_live_ready_preview();
    test_scheduler_cancel_clears_place_latch_before_another_preview();
    test_scheduler_reset_clears_place_latch_before_another_preview();
    test_scheduler_clears_far_place_when_carry_authority_ends();
    test_frame_handoff_first_owned_frame_is_exact_displayed_reference();
    test_frame_handoff_solves_positive_targeted_constraint_on_owned_entry();
    test_calibrated_bridge_keeps_exact_25_hz_handoff_continuous_and_reaches_oracle();
    test_frame_handoff_applies_validated_semantic_hand_constraint_and_releases_from_it();
    test_frame_handoff_starts_matching_constraint_after_attachment_inside_owned_epoch();
    test_delayed_constraint_rejects_free_lifecycle();
    test_delayed_constraint_accepts_targeted_lifecycle();
    test_delayed_constraint_rejects_identity_switched_before_first_constraint();
    test_hand_constraint_keeps_layered_carry_lower_body_exact_and_selected_only();
    test_layered_carry_final_composite_guard_is_atomic_for_both_hands();
    test_hand_constraint_mismatch_or_invalid_input_disables_atomically();
    test_hand_constraint_reset_and_reentry_recalibrates_selected_hand();
    test_frame_handoff_crosses_179_9_to_180_1_incrementally();
    test_frame_handoff_keeps_captured_neck_and_head_while_owned();
    test_hold_to_layered_carry_first_frame_preserves_rendered_lower_body();
    test_layered_carry_lower_body_rebases_moving_target_and_cleans_up();
    test_layered_carry_deadline_target_change_rebases_without_exact_snap();
    test_layered_carry_terminal_requires_velocity_offsets_to_decay();
    test_layered_carry_terminal_requires_position_and_rotation_offsets_to_decay();
    test_layered_carry_terminal_accepts_tightly_bounded_moving_target();
    test_layered_carry_keeps_fresh_lower_body_and_contacts_bit_exact();
    test_layered_carry_released_inactive_arm_uses_fresh_locomotion_authority();
    test_inactive_arm_locomotion_intent_does_not_change_other_carry_modes();
    test_inactive_arm_release_bounds_moving_target_and_spine();
    test_inactive_arm_return_to_recorded_target_is_bounded_and_convergent();
    test_frame_handoff_release_is_continuous_and_relinquishes_after_blend();
    test_all_place_states_keep_one_full_body_runtime_ownership_epoch();
    test_place_release_generation_change_ends_constraint_without_free_object_solve();
    test_normal_release_starts_at_last_displayed_place_release_pose();
    test_place_release_active_arm_return_is_bounded_after_ownership();
    test_same_generation_free_lifecycle_ends_constraint_without_resurrection();
    test_left_place_release_reset_clears_blend_and_rejects_free_entry_solve();
    test_place_release_clears_stale_inactive_arm_return_before_composite();
    test_frame_handoff_reset_and_reentry_capture_fresh_flat_reference();
    test_frame_handoff_preserves_fresh_complete_nonowned_pose_for_25_frames();
    test_frame_handoff_exposes_rendered_flat_root_sync();
    test_frame_handoff_keeps_layered_carry_simulation_root_live();
    test_scene_handoff_publishes_each_fresh_25_hz_sample_without_lag();
    test_scene_handoff_publishes_fresh_equal_plateau_samples_exactly();
    test_scene_handoff_publishes_fresh_rotation_and_holds_only_explicit_cache();
    test_scene_handoff_retains_post_failure_held_pose_until_registry_reclaims_authority();
    test_first_free_scene_sample_is_bit_exact_final_attached_runtime_pose();
    test_scene_handoff_rejects_invalid_alpha_atomically();
    test_carry_label_is_only_specific_during_carry();
    test_place_debug_status_prefers_active_controller_preview_values();
    test_demo_target_preserves_object_in_table_transform();
    test_demo_scene_rejects_every_malformed_consumed_shape_as_format_error();
    test_demo_destination_rejects_unsupported_fit_categories();
    test_cross_pack_frame_count_validation();
    test_debug_draw_uses_real_correction_geometry_and_complete_text();
    test_controller_and_make_clock_policy();
    test_controller_smart_pickup_two_phase_production_seam();
    return 0;
}
