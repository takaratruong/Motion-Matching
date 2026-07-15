#include "interaction_controller_adapter.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using interaction::ControllerInteractionAdapter;
using interaction::ControllerInteractionEdges;
using interaction::ControllerInteractionFrameHandoff;
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
using interaction::PickRequest;
using interaction::Pose;
using interaction::Reason;
using interaction::ResultCode;
using interaction::RuntimeInput;
using interaction::RuntimeOutput;
using interaction::RuntimeState;
using interaction::TargetHandle;
using interaction::Transform;

struct FlatWorldPose {
    std::array<vec3, interaction::kFlatControllerBoneCount> positions{};
    std::array<vec3, interaction::kFlatControllerBoneCount> velocities{};
    std::array<quat, interaction::kFlatControllerBoneCount> rotations{};
    std::array<vec3, interaction::kFlatControllerBoneCount>
        angular_velocities{};
};

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
        left_diagnostics.attached != right_diagnostics.attached ||
        left_diagnostics.recorded_carry != right_diagnostics.recorded_carry ||
        left_diagnostics.pack_available != right_diagnostics.pack_available) {
        return false;
    }
    return true;
}

bool near(float left, float right, float tolerance = 1.0e-5F) {
    return std::fabs(left - right) <= tolerance;
}

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
    output.diagnostics.attached = true;
    output.diagnostics.recorded_carry = true;
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

void test_exact_constants() {
    static_assert(
        interaction::kControllerStepSeconds == 1.0F / 60.0F,
        "controller step must be exact binary32 1/60");
    static_assert(
        interaction::kInteractionRuntimeStepSeconds == 1.0F / 25.0F,
        "runtime step must be exact binary32 1/25");
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

void test_flat_bridge_collapse_round_trips_all_channels_and_contacts() {
    const FlatControllerPose original = make_flat_pose();
    const Pose reference = make_pose(0.5F);
    const Pose expanded =
        interaction::expand_flat_controller_pose(original, reference);
    const FlatControllerPose collapsed =
        interaction::collapse_interaction_pose(expanded, original);
    assert_flat_pose_near(collapsed, original);

    for (size_t bone : {13U, 14U}) {
        assert(vec_bits_equal(
            collapsed.positions[bone], original.positions[bone]));
        assert(vec_bits_equal(
            collapsed.velocities[bone], original.velocities[bone]));
        assert(quat_bits_equal(
            collapsed.rotations[bone], original.rotations[bone]));
        assert(vec_bits_equal(
            collapsed.angular_velocities[bone],
            original.angular_velocities[bone]));
    }
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
            antipodal_expanded, antipodal),
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
    expect_format_error([&] {
        (void)interaction::collapse_interaction_pose(
            invalid_interaction, fallback);
    });
    assert(pose_bits_equal(invalid_interaction, invalid_interaction_before));
    assert(flat_pose_bits_equal(fallback, fallback_before));

    invalid_interaction = make_pose(1.75F);
    fallback.velocities[18].y =
        std::numeric_limits<float>::quiet_NaN();
    const FlatControllerPose invalid_fallback_before = fallback;
    expect_format_error([&] {
        (void)interaction::collapse_interaction_pose(
            invalid_interaction, fallback);
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
        expanded, input.pose);
    assert(input.before == kBefore);
    assert(input.after == kAfter);
    assert(output.before == kBefore);
    assert(output.after == kAfter);
    assert_flat_pose_near(output.pose, input.pose);
}

void test_scheduler_cadence_and_cache() {
    ControllerInteractionScheduler scheduler;
    assert(scheduler.phase() == 0);

    int snapshot_calls = 0;
    int resolver_calls = 0;
    int update_calls = 0;
    std::vector<int> due_ticks;
    RuntimeOutput newest{};

    for (int tick = 1; tick <= 60; ++tick) {
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

        assert(update_calls - calls_before <= 1);
        if (update_calls == calls_before) {
            assert(output_fields_equal(observed, before));
        } else {
            assert(output_fields_equal(observed, newest));
        }
    }

    assert(update_calls == 25);
    assert(snapshot_calls == 25);
    assert(resolver_calls == 0);
    assert(scheduler.phase() == 0);
    const std::vector<int> expected_first_block = {3, 5, 8, 10, 12};
    assert(std::vector<int>(due_ticks.begin(), due_ticks.begin() + 5) ==
           expected_first_block);
    for (size_t block = 0; block < 5; ++block) {
        for (size_t index = 0; index < expected_first_block.size(); ++index) {
            assert(due_ticks[block * 5 + index] ==
                   static_cast<int>(block * 12) + expected_first_block[index]);
        }
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
    scheduler.tick({true, true, false}, snapshot_provider, resolver, update);
    assert(snapshot_calls == 0);
    assert(resolver_calls == 0);
    assert(update_calls == 0);

    scheduler.tick({false, false, true}, snapshot_provider, resolver, update);
    assert(snapshot_calls == 1);
    assert(resolver_calls == 1);
    assert(update_calls == 1);
    assert(delivered[0].interact_pressed);
    assert(delivered[0].cancel_pressed);
    assert(delivered[0].reset_pressed);
    assert(delivered[0].pick_request.has_value());
    assert(delivered[0].pick_request->target == request.target);
    assert(delivered[0].pick_request->affordance_id == request.affordance_id);
    assert(delivered[0].pick_request->request_id == request.request_id);

    scheduler.tick({}, snapshot_provider, resolver, update);
    scheduler.tick({}, snapshot_provider, resolver, update);
    assert(snapshot_calls == 2);
    assert(resolver_calls == 1);
    assert(update_calls == 2);
    assert(!delivered[1].interact_pressed);
    assert(!delivered[1].cancel_pressed);
    assert(!delivered[1].reset_pressed);
    assert(!delivered[1].pick_request.has_value());

    ControllerInteractionScheduler due_edge_scheduler;
    due_edge_scheduler.tick({}, snapshot_provider, resolver, update);
    due_edge_scheduler.tick({}, snapshot_provider, resolver, update);
    due_edge_scheduler.tick(
        {true, true, true}, snapshot_provider, resolver, update);
    const RuntimeInput& due_edge_input = delivered.back();
    assert(due_edge_input.interact_pressed);
    assert(due_edge_input.cancel_pressed);
    assert(due_edge_input.reset_pressed);
}

void test_cache_changes_only_after_successful_due_delivery() {
    ControllerInteractionScheduler scheduler;
    const auto snapshot_provider = [] { return make_snapshot(2.0F); };
    const auto resolver = [](const LocomotionSnapshot&) {
        return std::optional<PickRequest>{PickRequest{{8, 2}, 5, 99}};
    };
    int calls = 0;

    scheduler.tick({true, false, false}, snapshot_provider, resolver,
                   [&](const RuntimeInput&) {
                       ++calls;
                       return make_complete_output(1, RuntimeState::Locomotion);
                   });
    scheduler.tick({}, snapshot_provider, resolver,
                   [&](const RuntimeInput&) {
                       ++calls;
                       return make_complete_output(1, RuntimeState::Locomotion);
                   });
    assert(calls == 0);
    const RuntimeOutput initial{};
    assert(output_fields_equal(scheduler.cached_output(), initial));

    bool threw = false;
    try {
        scheduler.tick({}, snapshot_provider, resolver,
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
    assert(output_fields_equal(scheduler.cached_output(), initial));

    scheduler.tick({}, snapshot_provider, resolver,
                   [&](const RuntimeInput&) {
                       ++calls;
                       return make_complete_output(2, RuntimeState::Disabled);
                   });
    const RuntimeOutput held = scheduler.cached_output();
    scheduler.tick({}, snapshot_provider, resolver,
                   [&](const RuntimeInput& input) {
                       ++calls;
                       assert(input.interact_pressed);
                       return make_complete_output(3, RuntimeState::Locomotion);
                   });
    assert(calls == 2);
    assert(!output_fields_equal(scheduler.cached_output(), held));
}

void test_adapter_blends_every_pose_channel_and_contacts() {
    ControllerInteractionAdapter adapter;
    const Pose locomotion = make_pose(-2.0F);
    RuntimeOutput idle;
    const Pose idle_output =
        adapter.apply(locomotion, idle, interaction::kControllerStepSeconds);
    assert(pose_bits_equal(idle_output, locomotion));

    RuntimeOutput owned;
    owned.owns_pose = true;
    owned.pose = make_pose(4.0F);
    const Pose first =
        adapter.apply(locomotion, owned, interaction::kControllerStepSeconds);
    assert(!pose_bits_equal(first, locomotion));
    assert(!pose_bits_equal(first, owned.pose));

    const float alpha = interaction::kControllerStepSeconds / 0.25F;
    assert_vec_near(
        first.positions[5],
        lerp(locomotion.positions[5], owned.pose.positions[5], alpha));
    assert_vec_near(
        first.velocities[6],
        lerp(locomotion.velocities[6], owned.pose.velocities[6], alpha));
    assert_same_rotation(
        first.rotations[7],
        quat_nlerp_shortest(
            locomotion.rotations[7], owned.pose.rotations[7], alpha));
    assert_vec_near(
        first.angular_velocities[8],
        lerp(
            locomotion.angular_velocities[8],
            owned.pose.angular_velocities[8],
            alpha));
    assert(near(
        first.hand_dof[4],
        lerpf(locomotion.hand_dof[4], owned.pose.hand_dof[4], alpha)));
    assert(near(
        first.hand_dof_velocities[9],
        lerpf(
            locomotion.hand_dof_velocities[9],
            owned.pose.hand_dof_velocities[9],
            alpha)));

    Pose blended = first;
    for (int tick = 2; tick <= 7; ++tick) {
        blended =
            adapter.apply(locomotion, owned, interaction::kControllerStepSeconds);
        assert(blended.foot_contacts == locomotion.foot_contacts);
    }
    blended = adapter.apply(
        locomotion, owned, interaction::kControllerStepSeconds);
    assert(blended.foot_contacts == owned.pose.foot_contacts);

    for (int tick = 9; tick <= 15; ++tick) {
        blended =
            adapter.apply(locomotion, owned, interaction::kControllerStepSeconds);
    }
    assert(pose_bits_equal(blended, owned.pose));
    const Pose after_settlement =
        adapter.apply(locomotion, owned, interaction::kControllerStepSeconds);
    assert(pose_bits_equal(after_settlement, owned.pose));
}

void test_adapter_shortest_antipodal_rotation() {
    ControllerInteractionAdapter adapter;
    Pose source{};
    Pose target{};
    const quat positive_target =
        quat_from_angle_axis(0.75F, vec3(0.0F, 1.0F, 0.0F));
    target.rotations.fill(-positive_target);
    RuntimeOutput owned;
    owned.owns_pose = true;
    owned.pose = target;

    const Pose first =
        adapter.apply(source, owned, interaction::kControllerStepSeconds);
    for (quat rotation : first.rotations) {
        assert(near(quat_length(rotation), 1.0F, 2.0e-5F));
        assert(rotation.w > 0.0F);
        assert(rotation.y > 0.0F);
    }
}

void test_adapter_exit_reentry_and_reset_capture_fresh_locomotion() {
    ControllerInteractionAdapter adapter;
    RuntimeOutput owned;
    owned.owns_pose = true;
    owned.pose = make_pose(9.0F);

    const Pose first_source = make_pose(1.0F);
    (void)adapter.apply(
        first_source, owned, interaction::kControllerStepSeconds);

    RuntimeOutput idle;
    const Pose current_locomotion = make_pose(3.0F);
    const Pose exited = adapter.apply(
        current_locomotion, idle, interaction::kControllerStepSeconds);
    assert(pose_bits_equal(exited, current_locomotion));

    const Pose reentered = adapter.apply(
        current_locomotion, owned, interaction::kControllerStepSeconds);
    const float alpha = interaction::kControllerStepSeconds / 0.25F;
    assert_vec_near(
        reentered.positions[0],
        lerp(
            current_locomotion.positions[0],
            owned.pose.positions[0],
            alpha));

    adapter.reset();
    const Pose reset_source = make_pose(5.0F);
    const Pose after_reset = adapter.apply(
        reset_source, owned, interaction::kControllerStepSeconds);
    assert_vec_near(
        after_reset.positions[0],
        lerp(reset_source.positions[0], owned.pose.positions[0], alpha));
}

void test_frame_handoff_preserves_fresh_complete_nonowned_pose_for_60_frames() {
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
    Pose previous{};

    for (int tick = 1; tick <= 60; ++tick) {
        const Pose fresh = make_pose(20.0F + static_cast<float>(tick));
        if (tick != 1) {
            assert(!pose_bits_equal(fresh, previous));
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
        assert(pose_bits_equal(disabled_frame.pose, fresh));
        assert(pose_bits_equal(loaded_frame.pose, fresh));
        assert(!disabled_frame.owns_pose);
        assert(!loaded_frame.owns_pose);
        assert(!disabled_frame.synchronize_simulation_root);
        assert(!loaded_frame.synchronize_simulation_root);
    }

    assert(disabled_runtime_calls == 25);
    assert(loaded_runtime_calls == 25);
    assert(disabled_scheduler.phase() == 0);
    assert(loaded_scheduler.phase() == 0);
}

void test_frame_handoff_retains_channels_and_exposes_root_sync() {
    for (float value : interaction::kFlatControllerRestHandDof) {
        assert(float_bits_equal(value, 0.0F));
    }
    for (float value : interaction::kFlatControllerRestHandDofVelocities) {
        assert(float_bits_equal(value, 0.0F));
    }

    ControllerInteractionFrameHandoff handoff;
    Pose locomotion = make_pose(-4.0F);
    locomotion.hand_dof = interaction::kFlatControllerRestHandDof;
    locomotion.hand_dof_velocities =
        interaction::kFlatControllerRestHandDofVelocities;
    RuntimeOutput owned;
    owned.owns_pose = true;
    owned.pose = make_pose(8.0F, true);

    ControllerInteractionFrameState frame{};
    for (int tick = 1; tick <= 15; ++tick) {
        frame = handoff.apply(
            locomotion, owned, interaction::kControllerStepSeconds);
        assert(frame.owns_pose);
        assert(frame.synchronize_simulation_root);
        assert(vec_bits_equal(
            frame.simulation_root_position, frame.pose.positions[0]));
        assert(quat_bits_equal(
            frame.simulation_root_rotation, frame.pose.rotations[0]));
    }
    assert(pose_bits_equal(frame.pose, owned.pose));
    assert(frame.pose.hand_dof == owned.pose.hand_dof);
    assert(frame.pose.hand_dof_velocities ==
           owned.pose.hand_dof_velocities);
    assert(frame.pose.foot_contacts == owned.pose.foot_contacts);

    vec3 simulation_position(91.0F, 92.0F, 93.0F);
    quat simulation_rotation = quat_from_angle_axis(
        0.3F, vec3(0.0F, 1.0F, 0.0F));
    if (frame.synchronize_simulation_root) {
        simulation_position = frame.simulation_root_position;
        simulation_rotation = frame.simulation_root_rotation;
    }
    assert(vec_bits_equal(simulation_position, frame.pose.positions[0]));
    assert(quat_bits_equal(simulation_rotation, frame.pose.rotations[0]));

    const Pose fresh_nonowned = make_pose(31.0F);
    RuntimeOutput nonowned;
    const ControllerInteractionFrameState released = handoff.apply(
        fresh_nonowned, nonowned, interaction::kControllerStepSeconds);
    assert(pose_bits_equal(released.pose, fresh_nonowned));
    assert(!released.synchronize_simulation_root);
}

void test_frame_handoff_keeps_layered_carry_simulation_root_live() {
    const Pose locomotion = make_pose(-4.0F);
    RuntimeOutput output;
    output.owns_pose = true;
    output.pose = make_pose(8.0F, true);

    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = false;
    ControllerInteractionFrameHandoff layered_handoff;
    const ControllerInteractionFrameState layered = layered_handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    assert(layered.owns_pose);
    assert(!layered.synchronize_simulation_root);

    output.diagnostics.state = RuntimeState::Align;
    ControllerInteractionFrameHandoff pre_carry_handoff;
    const ControllerInteractionFrameState pre_carry =
        pre_carry_handoff.apply(
            locomotion, output, interaction::kControllerStepSeconds);
    assert(pre_carry.synchronize_simulation_root);

    output.diagnostics.state = RuntimeState::Carry;
    output.diagnostics.recorded_carry = true;
    ControllerInteractionFrameHandoff recorded_handoff;
    const ControllerInteractionFrameState recorded = recorded_handoff.apply(
        locomotion, output, interaction::kControllerStepSeconds);
    assert(recorded.synchronize_simulation_root);
}

void test_scene_handoff_retains_attached_pose_until_registry_reclaims_authority() {
    ControllerInteractionSceneHandoff handoff;
    InteractionTarget target;
    target.handle = {41, 3};
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
        &target, attached, authored_fallback);
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
        &target, post_failure, authored_fallback);
    assert(held_scene.runtime_authority);
    assert(transform_bits_equal(
        held_scene.object_world, post_failure.object_world));

    InteractionTarget replacement = target;
    replacement.handle = {41, 9};
    replacement.state = ObjectState::Free;
    replacement.object_world = {
        vec3(-3.0F, 0.9F, 4.0F),
        quat_from_angle_axis(-0.5F, vec3(0.0F, 1.0F, 0.0F))};
    const ControllerInteractionSceneState free_scene = handoff.apply(
        &replacement, post_failure, authored_fallback);
    assert(!free_scene.runtime_authority);
    assert(transform_bits_equal(
        free_scene.object_world, replacement.object_world));

    replacement.state = ObjectState::Targeted;
    replacement.object_world.position.x += 0.25F;
    const ControllerInteractionSceneState targeted_scene = handoff.apply(
        &replacement, post_failure, authored_fallback);
    assert(!targeted_scene.runtime_authority);
    assert(transform_bits_equal(
        targeted_scene.object_world, replacement.object_world));
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
    database.range_starts = {0};
    database.range_stops = {4};
    database.phases = {
        static_cast<uint8_t>(Phase::Approach),
        static_cast<uint8_t>(Phase::Reach),
        static_cast<uint8_t>(Phase::Contact),
        static_cast<uint8_t>(Phase::Lift)};
    database.active_hands = {static_cast<uint8_t>(Hand::Left)};

    const Transform source_table{
        vec3(2.0F, 0.8F, -1.0F),
        quat_from_angle_axis(0.65F, vec3(0.0F, 1.0F, 0.0F))};
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
        const Transform object = frame == 1
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

    const InteractionTarget target =
        interaction::make_controller_demo_target(database);
    assert(target.handle.id != 0U);
    assert(target.handle.generation != 0U);
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
    assert(target.state == ObjectState::Free);

    const Transform selected_source = transform_from_arrays(
        database.object_positions, database.object_rotations, 1);
    assert_vec_near(selected_source.position, source_object.position);
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
    const std::string debug = read_text("interaction_debug_draw.h");
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
}

void test_controller_and_make_clock_policy() {
    const std::string controller = read_text("controller.cpp");
    assert(controller.find("SetTargetFPS(60);") != std::string::npos);
    assert(controller.find(
               "const float dt = interaction::kControllerStepSeconds;") !=
           std::string::npos);
    assert(controller.find("ControllerInteractionScheduler") !=
           std::string::npos);
    assert(controller.find("interaction_scheduler.tick(") != std::string::npos);
    assert(controller.find("interaction_frame_handoff.apply(") !=
           std::string::npos);
    assert(controller.find("interaction_scene_handoff.apply(") !=
           std::string::npos);
    assert(controller.find("interaction_registry.find_by_id(") !=
           std::string::npos);
    assert(controller.find("++next_handle.generation") == std::string::npos);
    assert(controller.find("interaction::kControllerStepSeconds") !=
           std::string::npos);
    assert(controller.find("GetFrameTime(") == std::string::npos);
    assert(controller.find("runtime_accumulator") == std::string::npos);
    assert(controller.find("runtime_output_alpha") == std::string::npos);
    assert(controller.find(
               "emscripten_set_main_loop_arg(update_callback, &u, 60, 1);") !=
           std::string::npos);
    assert(controller.find(
               "emscripten_set_main_loop_arg(update_callback, &u, 0, 1);") ==
           std::string::npos);
    assert(controller.find("InteractionRuntime::disabled(") !=
           std::string::npos);
    assert(controller.find("catch (const interaction::FormatError&") !=
           std::string::npos);

    const std::string makefile = read_text("Makefile");
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

int main() {
    test_exact_constants();
    test_flat_bridge_exact_parent_tree_anchor_map_and_unmapped_head();
    test_flat_bridge_expansion_matches_every_world_anchor_and_retains_reference();
    test_flat_bridge_collapse_round_trips_all_channels_and_contacts();
    test_flat_bridge_accepts_antipodal_equivalent_rotations();
    test_flat_bridge_rejects_invalid_input_without_mutation();
    test_flat_bridge_canaries_prove_exact_23_element_bounds();
    test_scheduler_cadence_and_cache();
    test_edges_latch_coalesce_and_clear_after_delivery();
    test_cache_changes_only_after_successful_due_delivery();
    test_adapter_blends_every_pose_channel_and_contacts();
    test_adapter_shortest_antipodal_rotation();
    test_adapter_exit_reentry_and_reset_capture_fresh_locomotion();
    test_frame_handoff_preserves_fresh_complete_nonowned_pose_for_60_frames();
    test_frame_handoff_retains_channels_and_exposes_root_sync();
    test_frame_handoff_keeps_layered_carry_simulation_root_live();
    test_scene_handoff_retains_attached_pose_until_registry_reclaims_authority();
    test_carry_label_is_only_specific_during_carry();
    test_demo_target_preserves_object_in_table_transform();
    test_cross_pack_frame_count_validation();
    test_debug_draw_uses_real_correction_geometry_and_complete_text();
    test_controller_and_make_clock_policy();
    return 0;
}
