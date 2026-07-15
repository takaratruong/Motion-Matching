#include "interaction_place.h"
#include "interaction_rotation_gate.h"
#include "g1_arm_joint_metadata.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iomanip>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace interaction;

constexpr float kPositionLimit = 0.12F;
constexpr float kOrientationLimit = 0.436332313F;
constexpr float kTenDegrees = 0.174532925F;
constexpr int32_t kFrameCount = 30;
constexpr int32_t kContactFrame = 8;
constexpr int32_t kLiftFrame = 12;
constexpr int32_t kHoldFrame = 18;
constexpr int32_t kStableWindowStop = 23;
constexpr uint64_t kProfileId = 3001U;
constexpr size_t kRightHand = static_cast<size_t>(g1_skeleton::RightWrist);
constexpr size_t kRoot = static_cast<size_t>(g1_skeleton::Simulation);

int g_failures = 0;

#define TEST_CHECK(expression)                                                \
    do {                                                                      \
        if (!(expression)) {                                                  \
            std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: "  \
                      << #expression << '\n';                                 \
            ++g_failures;                                                     \
        }                                                                     \
    } while (false)

bool near(float left, float right, float tolerance = 1.0e-5F) {
    return std::abs(left - right) <= tolerance;
}

bool near(vec3 left, vec3 right, float tolerance = 1.0e-5F) {
    return near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near(quat left, quat right, float tolerance = 1.0e-5F) {
    return quat_angle_between(left, right) <= tolerance;
}

bool near(Transform left, Transform right, float tolerance = 1.0e-5F) {
    return near(left.position, right.position, tolerance) &&
           near(left.rotation, right.rotation, tolerance);
}

bool exact_ik_equal(const IKConfig& left, const IKConfig& right) {
    return left.maximum_request_position_m ==
               right.maximum_request_position_m &&
           left.maximum_request_orientation_radians ==
               right.maximum_request_orientation_radians &&
           left.accepted_position_m == right.accepted_position_m &&
           left.accepted_orientation_radians ==
               right.accepted_orientation_radians &&
           left.damping == right.damping &&
           left.finite_difference_radians ==
               right.finite_difference_radians &&
           left.orientation_scale_m_per_radian ==
               right.orientation_scale_m_per_radian &&
           left.maximum_step_radians == right.maximum_step_radians &&
           left.maximum_iterations == right.maximum_iterations;
}

template<class Exception, class Function>
bool throws_as(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

size_t vec_offset(int32_t frame, size_t bone) {
    return (static_cast<size_t>(frame) * g1_skeleton::BoneCount + bone) * 3U;
}

size_t quat_offset(int32_t frame, size_t bone) {
    return (static_cast<size_t>(frame) * g1_skeleton::BoneCount + bone) * 4U;
}

void write_vec(std::vector<float>& values, size_t index, vec3 value) {
    const size_t offset = index * 3U;
    values.at(offset) = value.x;
    values.at(offset + 1U) = value.y;
    values.at(offset + 2U) = value.z;
}

void write_quat(std::vector<float>& values, size_t index, quat value) {
    const size_t offset = index * 4U;
    values.at(offset) = value.w;
    values.at(offset + 1U) = value.x;
    values.at(offset + 2U) = value.y;
    values.at(offset + 3U) = value.z;
}

void write_bone_position(
    Database& database,
    int32_t frame,
    size_t bone,
    vec3 value) {
    const size_t offset = vec_offset(frame, bone);
    database.positions.at(offset) = value.x;
    database.positions.at(offset + 1U) = value.y;
    database.positions.at(offset + 2U) = value.z;
}

void write_bone_rotation(
    Database& database,
    int32_t frame,
    size_t bone,
    quat value) {
    const size_t offset = quat_offset(frame, bone);
    database.rotations.at(offset) = value.w;
    database.rotations.at(offset + 1U) = value.x;
    database.rotations.at(offset + 2U) = value.y;
    database.rotations.at(offset + 3U) = value.z;
}

Transform root_transform(const Pose& pose) {
    const WorldPose world = world_pose(pose);
    return {world.positions[kRoot], world.rotations[kRoot]};
}

Transform hand_transform(const Pose& pose, Hand hand = Hand::Right) {
    const WorldPose world = world_pose(pose);
    const size_t bone = hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : kRightHand;
    return {world.positions[bone], world.rotations[bone]};
}

PlacementSurface make_surface(
    uint64_t id = 900U,
    vec3 position = vec3(1.0F, 0.70F, 2.0F)) {
    PlacementSurface surface{};
    surface.handle = {id, 3U};
    surface.surface_world = {position, quat()};
    surface.support_volume_size = vec3(1.0F, 0.70F, 1.0F);
    surface.support_volume_world = {
        position - vec3(0.0F, 0.35F, 0.0F), quat()};
    surface.half_extent_x_m = 0.40F;
    surface.half_extent_z_m = 0.40F;
    surface.overhead_clearance_m = 0.50F;
    surface.affordances = {
        PlaceAffordance{
            77U,
            Transform{vec3(0.0F, 0.10F, 0.0F), quat()},
            vec3(0.0F, -0.10F, 0.0F),
            vec3(0.0F, 1.0F, 0.0F),
            0.01F,
        },
        PlaceAffordance{
            78U,
            Transform{vec3(0.20F, 0.10F, 0.0F), quat()},
            vec3(0.0F, -0.10F, 0.0F),
            vec3(0.0F, 1.0F, 0.0F),
            0.01F,
        },
    };
    return surface;
}

ObjectLocalBounds make_bounds() {
    return {vec3(), vec3(0.04F, 0.10F, 0.04F)};
}

Pose make_pose(vec3 root, Transform hand, float marker = 0.0F) {
    Pose pose{};
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        pose.rotations[bone] = quat();
        pose.velocities[bone] = vec3(
            0.10F + marker,
            0.20F + 0.5F * marker,
            0.30F - 0.25F * marker);
        pose.angular_velocities[bone] = vec3(
            0.40F - 0.25F * marker,
            0.50F + marker,
            0.60F + 0.5F * marker);
    }
    pose.positions[kRoot] = root;
    pose.positions[kRightHand] = hand.position - root;
    pose.rotations[kRightHand] = hand.rotation;
    for (size_t dof = 0; dof < pose.hand_dof.size(); ++dof) {
        pose.hand_dof[dof] = marker + 0.01F * static_cast<float>(dof);
        pose.hand_dof_velocities[dof] =
            0.70F + marker + 0.01F * static_cast<float>(dof);
    }
    pose.foot_contacts = {
        static_cast<uint8_t>(static_cast<int>(marker * 1000.0F) & 1),
        static_cast<uint8_t>(
            (static_cast<int>(marker * 1000.0F) + 1) & 1),
    };
    return pose;
}

Database make_reverse_database() {
    Database database{};
    database.version = 1U;
    database.endian_marker = 0x01020304U;
    database.fps_numerator = 25U;
    database.fps_denominator = 1U;
    database.frame_count = kFrameCount;
    database.bone_count = static_cast<uint32_t>(g1_skeleton::BoneCount);
    database.clip_count = 1U;
    database.hand_dof_count = 14U;
    database.parents.assign(
        g1_skeleton::kParents.begin(), g1_skeleton::kParents.end());
    database.range_starts = {0};
    database.range_stops = {kFrameCount};
    const size_t frames = static_cast<size_t>(kFrameCount);
    const size_t bones = g1_skeleton::BoneCount;
    database.positions.assign(frames * bones * 3U, 0.0F);
    database.velocities.assign(frames * bones * 3U, 0.0F);
    database.rotations.assign(frames * bones * 4U, 0.0F);
    database.angular_velocities.assign(frames * bones * 3U, 0.0F);
    database.foot_contacts.assign(frames * 2U, 0U);
    database.hand_contacts.assign(frames * 2U, 0U);
    database.hand_dof.assign(frames * 14U, 0.0F);
    database.hand_dof_velocities.assign(frames * 14U, 0.0F);
    database.phases.assign(frames, 0U);
    database.active_hands = {static_cast<uint8_t>(Hand::Right)};
    database.time_to_contact.assign(frames, 0.0F);
    database.object_positions.assign(frames * 3U, 0.0F);
    database.object_rotations.assign(frames * 4U, 0.0F);
    database.object_velocities.assign(frames * 3U, 0.0F);
    database.object_angular_velocities.assign(frames * 3U, 0.0F);
    database.table_positions.assign(3U, 0.0F);
    database.table_rotations.assign(4U, 0.0F);
    database.table_sizes.assign(3U, 1.0F);
    database.object_dimensions = {0.08F, 0.20F, 0.08F};
    database.grasp_positions_object.assign(3U, 0.0F);
    database.grasp_rotations_object.assign(4U, 0.0F);
    database.approach_directions_object = {0.0F, 1.0F, 0.0F};
    database.source_frames.resize(frames);
    database.table_rotations[0] = 1.0F;
    database.grasp_rotations_object[0] = 1.0F;

    for (int32_t frame = 0; frame < kFrameCount; ++frame) {
        Phase phase = Phase::Approach;
        if (frame >= 4 && frame < kContactFrame) phase = Phase::Reach;
        if (frame >= kContactFrame && frame < kLiftFrame) {
            phase = Phase::Contact;
        }
        if (frame >= kLiftFrame && frame < kHoldFrame) phase = Phase::Lift;
        if (frame >= kHoldFrame) phase = Phase::Hold;
        database.phases[static_cast<size_t>(frame)] =
            static_cast<uint8_t>(phase);
        database.source_frames[static_cast<size_t>(frame)] = frame;
        const vec3 root(0.01F * static_cast<float>(frame), 0.0F, 0.0F);
        const float object_height = frame <= kContactFrame
            ? 0.80F
            : std::min(
                  0.92F,
                  0.80F + 0.012F *
                      static_cast<float>(frame - kContactFrame));
        const Transform object{
            vec3(0.0F, object_height, 0.0F), quat()};
        Transform hand = object;
        if (frame == kHoldFrame) hand.position.x += 0.020001F;
        const Pose pose = make_pose(
            root, hand, 0.001F * static_cast<float>(frame));
        for (size_t bone = 0; bone < bones; ++bone) {
            write_bone_position(
                database, frame, bone, pose.positions[bone]);
            write_bone_rotation(
                database, frame, bone, pose.rotations[bone]);
            const size_t offset = vec_offset(frame, bone);
            database.velocities[offset] = pose.velocities[bone].x;
            database.velocities[offset + 1U] = pose.velocities[bone].y;
            database.velocities[offset + 2U] = pose.velocities[bone].z;
            database.angular_velocities[offset] =
                pose.angular_velocities[bone].x;
            database.angular_velocities[offset + 1U] =
                pose.angular_velocities[bone].y;
            database.angular_velocities[offset + 2U] =
                pose.angular_velocities[bone].z;
        }
        for (size_t dof = 0; dof < 14U; ++dof) {
            const size_t offset = static_cast<size_t>(frame) * 14U + dof;
            database.hand_dof[offset] = pose.hand_dof[dof];
            database.hand_dof_velocities[offset] =
                pose.hand_dof_velocities[dof];
        }
        database.foot_contacts[static_cast<size_t>(frame) * 2U] =
            pose.foot_contacts[0];
        database.foot_contacts[static_cast<size_t>(frame) * 2U + 1U] =
            pose.foot_contacts[1];
        if (frame >= kContactFrame && frame < kFrameCount - 1) {
            database.hand_contacts[static_cast<size_t>(frame) * 2U + 1U] = 1U;
        }
        write_vec(
            database.object_positions,
            static_cast<size_t>(frame),
            object.position);
        write_quat(
            database.object_rotations,
            static_cast<size_t>(frame),
            object.rotation);
        write_vec(
            database.object_velocities,
            static_cast<size_t>(frame),
            vec3(0.02F, 0.03F, 0.04F));
        write_vec(
            database.object_angular_velocities,
            static_cast<size_t>(frame),
            vec3(0.05F, 0.06F, 0.07F));
    }
    return database;
}

RecordedPlaceClip make_recorded_clip(uint64_t id = 101U) {
    RecordedPlaceClip clip{};
    clip.id = id;
    clip.object_profile_id = kProfileId;
    clip.fps_numerator = 25U;
    clip.fps_denominator = 1U;
    clip.entry_frame = 0;
    clip.commit_frame = 8;
    clip.release_frame = 12;
    clip.retract_stop_frame = 15;
    clip.hand = Hand::Right;
    clip.hand_in_object = {vec3(), quat()};
    clip.object_bounds = make_bounds();
    clip.source_surface = make_surface(501U, vec3(0.0F, 0.70F, 0.0F));
    clip.source_affordance_id = 77U;
    clip.poses.reserve(16U);
    clip.object_poses.reserve(16U);
    clip.active_hand_contacts.reserve(16U);
    for (int32_t frame = 0; frame < 16; ++frame) {
        const float height = frame <= clip.release_frame
            ? 0.80F + 0.01F * static_cast<float>(clip.release_frame - frame)
            : 0.80F;
        const Transform object{vec3(0.0F, height, 0.0F), quat()};
        clip.object_poses.push_back(object);
        clip.poses.push_back(make_pose(
            vec3(0.005F * static_cast<float>(frame), 0.0F, 0.0F),
            object,
            0.002F * static_cast<float>(frame)));
        clip.active_hand_contacts.push_back(
            frame <= clip.release_frame ? 1U : 0U);
    }
    return clip;
}

struct PlaceFixture {
    Database database{};
    PlaceMotionLibrary library{};
    PlaceMatchInput input{};
};

void refresh_pointers(PlaceFixture& fixture) {
    fixture.input.pickup_database = &fixture.database;
    fixture.input.library = &fixture.library;
}

PlaceFixture make_fixture(bool include_recorded = true) {
    PlaceFixture fixture{};
    fixture.database = make_reverse_database();
    if (include_recorded) fixture.library.recorded.push_back(make_recorded_clip());
    fixture.input.pickup_database = &fixture.database;
    fixture.input.library = &fixture.library;
    fixture.input.held_target = {42U, 7U};
    fixture.input.pickup_candidate.clip = 0;
    fixture.input.pickup_candidate.entry_frame = 0;
    fixture.input.pickup_candidate.contact_frame = kContactFrame;
    fixture.input.pickup_candidate.lift_frame = kLiftFrame;
    fixture.input.pickup_candidate.hold_frame = kHoldFrame;
    fixture.input.pickup_candidate.scene_from_source = {vec3(), quat()};
    fixture.input.pickup_candidate.group_costs = {1, 2, 3, 4, 5};
    fixture.input.held_object_profile_id = kProfileId;
    fixture.input.held_object_bounds = make_bounds();
    fixture.input.held_affordance = {
        7U,
        Hand::Right,
        Transform{vec3(), quat()},
        vec3(0.0F, 1.0F, 0.0F),
        0.01F,
    };
    fixture.input.surface = make_surface();
    fixture.input.place_affordance = fixture.input.surface.affordances.front();
    fixture.input.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    const RecordedPlaceClip reference = make_recorded_clip();
    const Transform source_release =
        reference.object_poses[static_cast<size_t>(reference.release_frame)];
    const Transform goal = placement_goal_world(
        fixture.input.surface,
        fixture.input.place_affordance.object_in_surface);
    const Transform scene = compose(goal, inverse(source_release));
    fixture.input.current_pose = reference.poses.front();
    fixture.input.current_pose.positions[kRoot] = compose(
        scene, root_transform(reference.poses.front())).position;
    fixture.input.current_pose.rotations[kRoot] = compose(
        scene, root_transform(reference.poses.front())).rotation;
    fixture.input.current_object_world = compose(
        scene, reference.object_poses.front());
    return fixture;
}

PlaceFixture copy_fixture(const PlaceFixture& source) {
    PlaceFixture copy = source;
    refresh_pointers(copy);
    return copy;
}

void remove_recorded(PlaceFixture& fixture) {
    fixture.library.recorded.clear();
    refresh_pointers(fixture);
}

void disable_reverse_tier(PlaceFixture& fixture) {
    const size_t active = static_cast<size_t>(Hand::Right);
    for (int32_t frame = kHoldFrame; frame < kFrameCount; ++frame) {
        fixture.database.hand_contacts[
            static_cast<size_t>(frame) * 2U + active] = 0U;
    }
    refresh_pointers(fixture);
}

void map_current_rigidly(PlaceMatchInput& input, Transform target_root) {
    const Transform current_root = root_transform(input.current_pose);
    const Transform mapping = compose(target_root, inverse(current_root));
    input.current_pose.positions[kRoot] = target_root.position;
    input.current_pose.rotations[kRoot] = target_root.rotation;
    input.current_object_world = compose(mapping, input.current_object_world);
}

PlaceMatchInput stage_carry_snapshot(
    PlaceMatchInput input,
    Transform staging_root_world) {
    map_current_rigidly(input, staging_root_world);
    return input;
}

float measured_yaw(quat rotation) {
    const vec3 forward = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

quat quaternion_with_measured_yaw(float target) {
    if (target == 0.0F) return quat();
    const float half = 0.5F * target;
    const float w = std::cos(half);
    float below = std::sin(half);
    float above = below;
    std::optional<quat> first_outside;
    for (int step = 0; step < 4096; ++step) {
        for (float y : {below, above}) {
            const quat rotation = quat_normalize(quat(w, 0.0F, y, 0.0F));
            const float measured = measured_yaw(rotation);
            if (measured == target) return rotation;
            if (!first_outside.has_value() &&
                ((target > 0.0F && measured > target) ||
                 (target < 0.0F && measured < target))) {
                first_outside = rotation;
            }
        }
        below = std::nextafter(
            below, -std::numeric_limits<float>::infinity());
        above = std::nextafter(
            above, std::numeric_limits<float>::infinity());
    }
    if (first_outside.has_value()) return *first_outside;
    throw std::logic_error("test could not encode measured yaw boundary");
}

void set_current_root_offset(
    PlaceMatchInput& input,
    Transform staging,
    float position_error,
    float yaw_error) {
    const quat yaw = quaternion_with_measured_yaw(yaw_error);
    const Transform target{
        staging.position + vec3(position_error, 0.0F, 0.0F),
        quat_normalize(quat_mul(yaw, staging.rotation)),
    };
    map_current_rigidly(input, target);
}

void set_current_hand_error(
    PlaceMatchInput& input,
    float position_error,
    float orientation_error) {
    const Transform current_root = root_transform(input.current_pose);
    map_current_rigidly(
        input, Transform{vec3(), current_root.rotation});
    input.current_pose.positions[kRightHand].x += position_error;
    input.current_pose.rotations[kRightHand] = quat_normalize(quat_mul(
        quat_from_angle_axis(
            orientation_error, vec3(0.0F, 1.0F, 0.0F)),
        input.current_pose.rotations[kRightHand]));
}

quat normalized_for_test(quat value) {
    const double norm = std::sqrt(
        static_cast<double>(value.w) * value.w +
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z);
    return value * static_cast<float>(1.0 / norm);
}

double production_relative_rotation_error(quat hand, quat object) {
    const quat object_rotation = normalized_for_test(object);
    const quat relative = normalized_for_test(quat_mul(
        quat_inv(object_rotation), normalized_for_test(hand)));
    const quat actual = normalized_for_test(relative);
    const quat expected = normalized_for_test(quat());
    const rotation_gate::Measure measured = rotation_gate::measure(
        actual, expected);
    return measured.valid
        ? measured.radians
        : std::numeric_limits<double>::max();
}

void set_current_hand_orientation_boundary(
    PlaceMatchInput& input,
    float limit,
    bool over) {
    set_current_hand_error(input, 0.0F, 0.0F);
    input.current_object_world.rotation = quat();
    const float half = 0.5F * limit;
    const float w = std::cos(half);
    float below_y = std::sin(half);
    float above_y = below_y;
    std::optional<quat> best_below;
    std::optional<quat> best_above;
    double best_below_error = -1.0;
    double best_above_error = std::numeric_limits<double>::infinity();
    for (int step = 0; step < 4096; ++step) {
        for (float y : {below_y, above_y}) {
            const quat candidate(w, 0.0F, y, 0.0F);
            Pose trial_pose = input.current_pose;
            trial_pose.rotations[kRightHand] = candidate;
            const double error = production_relative_rotation_error(
                hand_transform(trial_pose).rotation,
                input.current_object_world.rotation);
            if (error <= limit && error > best_below_error) {
                best_below = candidate;
                best_below_error = error;
            }
            if (error > limit && error < best_above_error) {
                best_above = candidate;
                best_above_error = error;
            }
        }
        below_y = std::nextafter(
            below_y, -std::numeric_limits<float>::infinity());
        above_y = std::nextafter(
            above_y, std::numeric_limits<float>::infinity());
    }
    const std::optional<quat>& selected = over ? best_above : best_below;
    if (!selected.has_value()) {
        throw std::logic_error("test could not encode orientation boundary");
    }
    input.current_pose.rotations[kRightHand] = *selected;
}

void expect_reversed_fallback(PlaceFixture fixture) {
    refresh_pointers(fixture);
    const PlaceResult result = select_place_motion(fixture.input);
    TEST_CHECK(result.accepted);
    TEST_CHECK(result.candidate.mode == PlaceMotionMode::ReversedPickup);
}

void expect_rejected(PlaceFixture fixture) {
    refresh_pointers(fixture);
    TEST_CHECK(!select_place_motion(fixture.input).accepted);
    TEST_CHECK(!preview_place_motion(fixture.input).accepted);
}

void test_frozen_public_contract() {
    static_assert(static_cast<uint8_t>(PlaceMotionMode::None) == 0U);
    static_assert(static_cast<uint8_t>(PlaceMotionMode::RecordedPlace) == 1U);
    static_assert(static_cast<uint8_t>(PlaceMotionMode::ReversedPickup) == 2U);
    static_assert(static_cast<uint8_t>(PlacePhase::Align) == 0U);
    static_assert(static_cast<uint8_t>(PlacePhase::Lower) == 1U);
    static_assert(static_cast<uint8_t>(PlacePhase::Release) == 2U);
    static_assert(static_cast<uint8_t>(PlacePhase::Retract) == 3U);
    static_assert(static_cast<uint8_t>(PlacePhase::Finished) == 4U);
    static_assert(std::is_same_v<
        decltype(&select_place_motion),
        PlaceResult (*)(const PlaceMatchInput&)>);
    static_assert(std::is_same_v<
        decltype(&preview_place_motion),
        PlaceStagingPreview (*)(const PlaceMatchInput&)>);
    static_assert(std::is_same_v<
        decltype(&PlacePlayer::start),
        void (PlacePlayer::*)(const PlaceCandidate&, const PlaceMatchInput&)>);
    static_assert(std::is_same_v<
        decltype(&PlacePlayer::advance),
        void (PlacePlayer::*)(float)>);
    static_assert(std::is_same_v<
        decltype(&PlacePlayer::sample),
        PlaceSample (PlacePlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&PlacePlayer::source_frame),
        int32_t (PlacePlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&PlacePlayer::phase),
        PlacePhase (PlacePlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&PlacePlayer::committed),
        bool (PlacePlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&PlacePlayer::release_due),
        bool (PlacePlayer::*)() const>);
    static_assert(std::is_same_v<
        decltype(&PlacePlayer::acknowledge_release),
        void (PlacePlayer::*)()>);
    static_assert(std::is_same_v<
        decltype(&PlacePlayer::finished),
        bool (PlacePlayer::*)() const>);

    const PlaceTimingConfig timing{};
    TEST_CHECK(timing.canonical_fps == 25.0F);
    TEST_CHECK(timing.playback_speed == 1.0F);
    TEST_CHECK(timing.entry_blend_seconds == 0.25F);
    TEST_CHECK(timing.reversed_commit_seconds == 0.50F);
    TEST_CHECK(timing.maximum_alignment_seconds == 1.00F);
    const PlaceMatchConfig match{};
    TEST_CHECK(match.maximum_entry_root_error_m == 0.25F);
    TEST_CHECK(match.maximum_entry_yaw_error_radians == 0.436332313F);

    const RecordedPlaceClip recorded{};
    TEST_CHECK(recorded.id == 0U);
    TEST_CHECK(recorded.entry_frame == -1);
    TEST_CHECK(recorded.commit_frame == -1);
    TEST_CHECK(recorded.release_frame == -1);
    TEST_CHECK(recorded.retract_stop_frame == -1);
    const PlaceCandidate candidate{};
    TEST_CHECK(candidate.mode == PlaceMotionMode::None);
    TEST_CHECK(candidate.selection_id == 0U);
    TEST_CHECK(candidate.direction == 0);
    const PlaceSample sample{};
    TEST_CHECK(sample.source_frame == -1);
    TEST_CHECK(sample.phase == PlacePhase::Align);
    TEST_CHECK(!sample.committed);
    const PlaceStagingPreview preview{};
    TEST_CHECK(!preview.accepted);
    TEST_CHECK(!preview.ready);
    TEST_CHECK(preview.ik_config_fingerprint == 0U);

    const PlacePlayer unstarted;
    TEST_CHECK(unstarted.finished());
    TEST_CHECK(unstarted.source_frame() == -1);
    TEST_CHECK(!unstarted.committed());
    TEST_CHECK(!unstarted.release_due());
    TEST_CHECK(throws_as<std::logic_error>([&] { (void)unstarted.sample(); }));
    TEST_CHECK(throws_as<std::logic_error>([&] { (void)unstarted.phase(); }));
}

void test_recorded_priority_and_dynamic_readiness() {
    PlaceFixture fixture = make_fixture();
    const PlaceResult preferred = select_place_motion(fixture.input);
    TEST_CHECK(preferred.accepted);
    TEST_CHECK(preferred.reason == Reason::None);
    TEST_CHECK(preferred.candidate.mode == PlaceMotionMode::RecordedPlace);
    TEST_CHECK(preferred.candidate.source_id == 101U);
    TEST_CHECK(preferred.candidate.selection_id != 0U);
    TEST_CHECK(preferred.candidate.commit_frame == 8);
    TEST_CHECK(preferred.candidate.direction == 1);
    TEST_CHECK(exact_ik_equal(preferred.candidate.ik, fixture.input.ik));

    PlaceFixture far = copy_fixture(fixture);
    const PlaceStagingPreview initial = preview_place_motion(far.input);
    TEST_CHECK(initial.accepted);
    set_current_root_offset(
        far.input, initial.staging_root_world, 0.80F, 0.60F);
    const PlaceStagingPreview far_preview = preview_place_motion(far.input);
    TEST_CHECK(far_preview.accepted);
    TEST_CHECK(!far_preview.ready);
    TEST_CHECK(far_preview.candidate.mode == PlaceMotionMode::RecordedPlace);
    TEST_CHECK(far_preview.candidate.source_id == 101U);
    TEST_CHECK(far_preview.candidate.selection_id != 0U);
    TEST_CHECK(exact_ik_equal(far_preview.ik, far.input.ik));
    TEST_CHECK(exact_ik_equal(far_preview.candidate.ik, far.input.ik));
    TEST_CHECK(far_preview.ik_config_fingerprint != 0U);

    PlaceMatchInput staged = stage_carry_snapshot(
        far.input, far_preview.staging_root_world);
    const PlaceStagingPreview staged_preview = preview_place_motion(staged);
    TEST_CHECK(staged_preview.accepted);
    TEST_CHECK(staged_preview.ready);
    TEST_CHECK(staged_preview.candidate.mode == PlaceMotionMode::RecordedPlace);
    TEST_CHECK(staged_preview.candidate.source_id ==
               far_preview.candidate.source_id);
    TEST_CHECK(staged_preview.candidate.selection_id !=
               far_preview.candidate.selection_id);
    TEST_CHECK(near(
        staged_preview.staging_root_world,
        far_preview.staging_root_world));
}

void assert_readiness_boundary(
    PlaceFixture fixture,
    PlaceMatchConfig config) {
    fixture.input.surface.surface_world.position.x = 0.0F;
    fixture.input.surface.surface_world.position.z = 0.0F;
    fixture.input.surface.support_volume_world.position.x = 0.0F;
    fixture.input.surface.support_volume_world.position.z = 0.0F;
    fixture.input.match = config;
    const PlaceStagingPreview base = preview_place_motion(fixture.input);
    TEST_CHECK(base.accepted);

    PlaceFixture root_boundary = copy_fixture(fixture);
    set_current_root_offset(
        root_boundary.input,
        base.staging_root_world,
        config.maximum_entry_root_error_m,
        0.0F);
    const PlaceStagingPreview exact_root =
        preview_place_motion(root_boundary.input);
    if (!exact_root.ready) {
        std::cerr << std::setprecision(10)
                  << "exact root diagnostics: " << exact_root.root_error_m
                  << " limit " << config.maximum_entry_root_error_m << '\n';
    }
    TEST_CHECK(exact_root.accepted);
    TEST_CHECK(exact_root.ready);
    TEST_CHECK(exact_root.candidate.source_id == base.candidate.source_id);
    TEST_CHECK(exact_root.candidate.mode == base.candidate.mode);

    PlaceFixture root_over = copy_fixture(fixture);
    set_current_root_offset(
        root_over.input,
        base.staging_root_world,
        std::nextafter(
            config.maximum_entry_root_error_m,
            std::numeric_limits<float>::infinity()),
        0.0F);
    const PlaceStagingPreview outside_root =
        preview_place_motion(root_over.input);
    if (outside_root.ready) {
        std::cerr << std::setprecision(10)
                  << "over root diagnostics: " << outside_root.root_error_m
                  << " limit " << config.maximum_entry_root_error_m << '\n';
    }
    TEST_CHECK(outside_root.accepted);
    TEST_CHECK(!outside_root.ready);
    TEST_CHECK(outside_root.candidate.source_id == base.candidate.source_id);
    TEST_CHECK(outside_root.candidate.mode == base.candidate.mode);

    PlaceFixture yaw_boundary = copy_fixture(fixture);
    set_current_root_offset(
        yaw_boundary.input,
        base.staging_root_world,
        0.0F,
        config.maximum_entry_yaw_error_radians);
    const PlaceStagingPreview exact_yaw =
        preview_place_motion(yaw_boundary.input);
    if (!exact_yaw.ready) {
        std::cerr << std::setprecision(10)
                  << "exact yaw diagnostics: " << exact_yaw.yaw_error_radians
                  << " limit " << config.maximum_entry_yaw_error_radians
                  << '\n';
    }
    TEST_CHECK(exact_yaw.accepted);
    TEST_CHECK(exact_yaw.ready);
    TEST_CHECK(exact_yaw.candidate.source_id == base.candidate.source_id);

    PlaceFixture yaw_over = copy_fixture(fixture);
    set_current_root_offset(
        yaw_over.input,
        base.staging_root_world,
        0.0F,
        std::nextafter(
            config.maximum_entry_yaw_error_radians,
            std::numeric_limits<float>::infinity()));
    const PlaceStagingPreview outside_yaw =
        preview_place_motion(yaw_over.input);
    TEST_CHECK(outside_yaw.accepted);
    TEST_CHECK(!outside_yaw.ready);
    TEST_CHECK(outside_yaw.candidate.source_id == base.candidate.source_id);
    TEST_CHECK(outside_yaw.candidate.mode == base.candidate.mode);
}

void test_readiness_boundaries_and_tighter_config() {
    assert_readiness_boundary(make_fixture(), PlaceMatchConfig{});
    PlaceMatchConfig tighter{};
    tighter.maximum_entry_root_error_m = 0.10F;
    tighter.maximum_entry_yaw_error_radians = 0.20F;
    assert_readiness_boundary(make_fixture(), tighter);
}

void test_invalid_timing_and_match_configs_reject_before_tiers() {
    using FloatMember = float PlaceTimingConfig::*;
    constexpr std::array<FloatMember, 5> timing_fields = {
        &PlaceTimingConfig::canonical_fps,
        &PlaceTimingConfig::playback_speed,
        &PlaceTimingConfig::entry_blend_seconds,
        &PlaceTimingConfig::reversed_commit_seconds,
        &PlaceTimingConfig::maximum_alignment_seconds,
    };
    for (FloatMember field : timing_fields) {
        for (float invalid : {
                 std::numeric_limits<float>::quiet_NaN(),
                 std::numeric_limits<float>::infinity()}) {
            PlaceFixture fixture = make_fixture();
            fixture.input.timing.*field = invalid;
            expect_rejected(std::move(fixture));
        }
    }
    const std::vector<std::function<void(PlaceTimingConfig&)>> timing_invalid = {
        [](PlaceTimingConfig& value) { value.canonical_fps = 24.999F; },
        [](PlaceTimingConfig& value) { value.playback_speed = 0.849F; },
        [](PlaceTimingConfig& value) { value.playback_speed = 1.151F; },
        [](PlaceTimingConfig& value) { value.entry_blend_seconds = 0.0F; },
        [](PlaceTimingConfig& value) { value.reversed_commit_seconds = 0.0F; },
        [](PlaceTimingConfig& value) { value.maximum_alignment_seconds = 0.0F; },
        [](PlaceTimingConfig& value) {
            value.maximum_alignment_seconds = 1.0001F;
        },
        [](PlaceTimingConfig& value) {
            value.maximum_alignment_seconds =
                value.entry_blend_seconds - 0.001F;
        },
        [](PlaceTimingConfig& value) {
            value.maximum_alignment_seconds =
                value.reversed_commit_seconds - 0.001F;
        },
    };
    for (const auto& mutate : timing_invalid) {
        PlaceFixture fixture = make_fixture();
        mutate(fixture.input.timing);
        expect_rejected(std::move(fixture));
    }

    using MatchMember = float PlaceMatchConfig::*;
    constexpr std::array<MatchMember, 2> match_fields = {
        &PlaceMatchConfig::maximum_entry_root_error_m,
        &PlaceMatchConfig::maximum_entry_yaw_error_radians,
    };
    for (MatchMember field : match_fields) {
        for (float invalid : {
                 std::numeric_limits<float>::quiet_NaN(),
                 std::numeric_limits<float>::infinity(),
                 0.0F,
                 -0.1F}) {
            PlaceFixture fixture = make_fixture();
            fixture.input.match.*field = invalid;
            expect_rejected(std::move(fixture));
        }
    }
    PlaceFixture root_over = make_fixture();
    root_over.input.match.maximum_entry_root_error_m = 0.250001F;
    expect_rejected(std::move(root_over));
    PlaceFixture yaw_over = make_fixture();
    yaw_over.input.match.maximum_entry_yaw_error_radians = 0.436333F;
    expect_rejected(std::move(yaw_over));
}

void test_ik_config_validation_and_authoritative_request_limits() {
    using IKFloatMember = float IKConfig::*;
    constexpr std::array<IKFloatMember, 8> fields = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (IKFloatMember field : fields) {
        for (float invalid : {
                 std::numeric_limits<float>::quiet_NaN(),
                 std::numeric_limits<float>::infinity()}) {
            PlaceFixture fixture = make_fixture();
            fixture.input.ik.*field = invalid;
            expect_rejected(std::move(fixture));
        }
    }
    constexpr std::array<IKFloatMember, 4> nonnegative = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
    };
    for (IKFloatMember field : nonnegative) {
        PlaceFixture fixture = make_fixture();
        fixture.input.ik.*field = -0.000001F;
        expect_rejected(std::move(fixture));
    }
    constexpr std::array<IKFloatMember, 4> positive = {
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (IKFloatMember field : positive) {
        PlaceFixture fixture = make_fixture();
        fixture.input.ik.*field = 0.0F;
        expect_rejected(std::move(fixture));
    }
    PlaceFixture negative_iterations = make_fixture();
    negative_iterations.input.ik.maximum_iterations = -1;
    expect_rejected(std::move(negative_iterations));
    PlaceFixture position_cap = make_fixture();
    position_cap.input.ik.maximum_request_position_m = 0.120001F;
    expect_rejected(std::move(position_cap));
    PlaceFixture orientation_cap = make_fixture();
    orientation_cap.input.ik.maximum_request_orientation_radians = 0.436333F;
    expect_rejected(std::move(orientation_cap));

    PlaceFixture exact_position = make_fixture();
    set_current_hand_error(exact_position.input, kPositionLimit, 0.0F);
    if (!select_place_motion(exact_position.input).accepted) {
        const Transform actual = hand_transform(exact_position.input.current_pose);
        const Transform expected = compose(
            exact_position.input.current_object_world,
            exact_position.input.held_affordance.hand_in_object);
        std::cerr << "exact IK position rejected reason "
                  << static_cast<int>(
                         select_place_motion(exact_position.input).reason)
                  << " actual error " << std::setprecision(10)
                  << length(actual.position - expected.position)
                  << '\n';
    }
    TEST_CHECK(select_place_motion(exact_position.input).accepted);
    PlaceFixture over_position = make_fixture();
    set_current_hand_error(
        over_position.input,
        std::nextafter(
            kPositionLimit, std::numeric_limits<float>::infinity()),
        0.0F);
    if (select_place_motion(over_position.input).accepted) {
        std::cerr << "over IK position unexpectedly accepted\n";
    }
    expect_rejected(std::move(over_position));

    PlaceFixture exact_orientation = make_fixture();
    set_current_hand_orientation_boundary(
        exact_orientation.input, kOrientationLimit, false);
    TEST_CHECK(select_place_motion(exact_orientation.input).accepted);
    PlaceFixture over_orientation = make_fixture();
    set_current_hand_orientation_boundary(
        over_orientation.input, kOrientationLimit, true);
    if (select_place_motion(over_orientation.input).accepted) {
        std::cerr << "over IK orientation unexpectedly accepted\n";
    }
    expect_rejected(std::move(over_orientation));

    PlaceFixture tighter_position = make_fixture();
    tighter_position.input.ik.maximum_request_position_m = 0.05F;
    set_current_hand_error(tighter_position.input, 0.05F, 0.0F);
    if (!select_place_motion(tighter_position.input).accepted) {
        const Transform actual = hand_transform(tighter_position.input.current_pose);
        const Transform expected = compose(
            tighter_position.input.current_object_world,
            tighter_position.input.held_affordance.hand_in_object);
        std::cerr << "tight IK position rejected reason "
                  << static_cast<int>(
                         select_place_motion(tighter_position.input).reason)
                  << " actual error " << std::setprecision(10)
                  << length(actual.position - expected.position)
                  << '\n';
    }
    TEST_CHECK(select_place_motion(tighter_position.input).accepted);
    PlaceFixture tighter_position_over = make_fixture();
    tighter_position_over.input.ik.maximum_request_position_m = 0.05F;
    set_current_hand_error(
        tighter_position_over.input, 0.050001F, 0.0F);
    if (select_place_motion(tighter_position_over.input).accepted) {
        std::cerr << "tight over IK position unexpectedly accepted\n";
    }
    expect_rejected(std::move(tighter_position_over));

    PlaceFixture tighter_orientation = make_fixture();
    tighter_orientation.input.ik.maximum_request_orientation_radians = 0.10F;
    set_current_hand_orientation_boundary(
        tighter_orientation.input, 0.10F, false);
    TEST_CHECK(select_place_motion(tighter_orientation.input).accepted);
    PlaceFixture tighter_orientation_over = make_fixture();
    tighter_orientation_over.input.ik.maximum_request_orientation_radians =
        0.10F;
    set_current_hand_orientation_boundary(
        tighter_orientation_over.input, 0.10F, true);
    if (select_place_motion(tighter_orientation_over.input).accepted) {
        std::cerr << "tight over IK orientation unexpectedly accepted\n";
    }
    expect_rejected(std::move(tighter_orientation_over));

    PlaceFixture zero_limits = make_fixture();
    zero_limits.input.ik.maximum_request_position_m = 0.0F;
    zero_limits.input.ik.maximum_request_orientation_radians = 0.0F;
    TEST_CHECK(select_place_motion(zero_limits.input).accepted);
}

void set_database_hand_world(
    Database& database,
    int32_t frame,
    Transform hand_world) {
    const Pose pose = pose_at_frame(database, frame);
    const Transform root = root_transform(pose);
    write_bone_position(
        database,
        frame,
        kRightHand,
        quat_mul_vec3(
            quat_inv(root.rotation), hand_world.position - root.position));
    write_bone_rotation(
        database,
        frame,
        kRightHand,
        quat_normalize(quat_mul(
            quat_inv(root.rotation), hand_world.rotation)));
}

Transform database_object_at(const Database& database, int32_t frame) {
    const size_t index = static_cast<size_t>(frame);
    const size_t position = index * 3U;
    const size_t rotation = index * 4U;
    return {
        vec3(
            database.object_positions.at(position),
            database.object_positions.at(position + 1U),
            database.object_positions.at(position + 2U)),
        quat(
            database.object_rotations.at(rotation),
            database.object_rotations.at(rotation + 1U),
            database.object_rotations.at(rotation + 2U),
            database.object_rotations.at(rotation + 3U)),
    };
}

void set_database_object_and_hand(
    Database& database,
    int32_t frame,
    Transform object) {
    write_vec(
        database.object_positions,
        static_cast<size_t>(frame),
        object.position);
    write_quat(
        database.object_rotations,
        static_cast<size_t>(frame),
        object.rotation);
    set_database_hand_world(database, frame, object);
}

void test_reverse_certification_uses_earliest_stable_prefix() {
    PlaceFixture fixture = make_fixture(false);
    const PlaceResult fallback = select_place_motion(fixture.input);
    TEST_CHECK(fallback.accepted);
    TEST_CHECK(fallback.candidate.mode == PlaceMotionMode::ReversedPickup);
    TEST_CHECK(fallback.candidate.entry_frame == kStableWindowStop);
    TEST_CHECK(fallback.candidate.entry_frame != kFrameCount - 1);
    TEST_CHECK(fallback.candidate.release_frame == kContactFrame);
    TEST_CHECK(fallback.candidate.stop_frame == 0);
    TEST_CHECK(fallback.candidate.direction == -1);
    TEST_CHECK(fallback.candidate.source_id != 0U);

    PlaceFixture late_change = copy_fixture(fixture);
    const size_t late = static_cast<size_t>(kFrameCount - 1);
    late_change.database.hand_contacts[late * 2U + 1U] = 0U;
    late_change.database.positions[vec_offset(kFrameCount - 1, kRightHand)] =
        std::numeric_limits<float>::quiet_NaN();
    late_change.database.object_positions[late * 3U] = 999.0F;
    const PlaceResult after_late_change =
        select_place_motion(late_change.input);
    TEST_CHECK(after_late_change.accepted);
    TEST_CHECK(after_late_change.candidate.entry_frame == kStableWindowStop);
    TEST_CHECK(after_late_change.candidate.selection_id ==
               fallback.candidate.selection_id);
    TEST_CHECK(after_late_change.candidate.source_id ==
               fallback.candidate.source_id);

    PlaceFixture boundary = copy_fixture(fixture);
    Transform hand = database_object_at(boundary.database, kHoldFrame);
    hand.position.x += 0.020F;
    set_database_hand_world(boundary.database, kHoldFrame, hand);
    const PlaceResult boundary_result = select_place_motion(boundary.input);
    TEST_CHECK(boundary_result.accepted);
    TEST_CHECK(boundary_result.candidate.entry_frame == kHoldFrame + 4);

    PlaceFixture rotation_boundary = copy_fixture(fixture);
    hand = database_object_at(rotation_boundary.database, kHoldFrame);
    hand.rotation = quat_from_angle_axis(
        kTenDegrees, vec3(0.0F, 1.0F, 0.0F));
    set_database_hand_world(rotation_boundary.database, kHoldFrame, hand);
    const PlaceResult rotation_boundary_result =
        select_place_motion(rotation_boundary.input);
    TEST_CHECK(rotation_boundary_result.accepted);
    TEST_CHECK(rotation_boundary_result.candidate.entry_frame ==
               kHoldFrame + 4);

    PlaceFixture rotation_over = copy_fixture(fixture);
    hand = database_object_at(rotation_over.database, kHoldFrame);
    hand.rotation = quat_from_angle_axis(
        kTenDegrees + 0.0000174533F,
        vec3(0.0F, 1.0F, 0.0F));
    set_database_hand_world(rotation_over.database, kHoldFrame, hand);
    const PlaceResult rotation_over_result =
        select_place_motion(rotation_over.input);
    TEST_CHECK(rotation_over_result.accepted);
    TEST_CHECK(rotation_over_result.candidate.entry_frame ==
               kStableWindowStop);
}

void append_unrelated_clip(Database& database) {
    const Database unrelated = make_reverse_database();
    const int32_t frame_offset = static_cast<int32_t>(database.frame_count);
    database.frame_count += unrelated.frame_count;
    ++database.clip_count;
    database.range_starts.push_back(frame_offset);
    database.range_stops.push_back(
        frame_offset + static_cast<int32_t>(unrelated.frame_count));
    database.active_hands.push_back(unrelated.active_hands.front());

    const auto append = [](auto& destination, const auto& source) {
        destination.insert(destination.end(), source.begin(), source.end());
    };
    append(database.positions, unrelated.positions);
    append(database.velocities, unrelated.velocities);
    append(database.rotations, unrelated.rotations);
    append(database.angular_velocities, unrelated.angular_velocities);
    append(database.foot_contacts, unrelated.foot_contacts);
    append(database.hand_contacts, unrelated.hand_contacts);
    append(database.hand_dof, unrelated.hand_dof);
    append(database.hand_dof_velocities, unrelated.hand_dof_velocities);
    append(database.phases, unrelated.phases);
    append(database.time_to_contact, unrelated.time_to_contact);
    append(database.object_positions, unrelated.object_positions);
    append(database.object_rotations, unrelated.object_rotations);
    append(database.object_velocities, unrelated.object_velocities);
    append(
        database.object_angular_velocities,
        unrelated.object_angular_velocities);
    append(database.table_positions, unrelated.table_positions);
    append(database.table_rotations, unrelated.table_rotations);
    append(database.table_sizes, unrelated.table_sizes);
    append(database.object_dimensions, unrelated.object_dimensions);
    append(
        database.grasp_positions_object,
        unrelated.grasp_positions_object);
    append(
        database.grasp_rotations_object,
        unrelated.grasp_rotations_object);
    append(
        database.approach_directions_object,
        unrelated.approach_directions_object);
    for (int32_t source : unrelated.source_frames) {
        database.source_frames.push_back(frame_offset + source);
    }
}

void test_reverse_identity_ignores_unconsulted_storage_and_clips() {
    PlaceFixture fixture = make_fixture(false);
    const PlaceResult baseline = select_place_motion(fixture.input);
    TEST_CHECK(baseline.accepted);
    TEST_CHECK(baseline.candidate.mode == PlaceMotionMode::ReversedPickup);

    PlaceFixture trailing_storage = copy_fixture(fixture);
    trailing_storage.database.positions.push_back(1234.0F);
    trailing_storage.database.hand_contacts.push_back(1U);
    trailing_storage.database.object_positions.push_back(-1234.0F);
    const PlaceResult after_storage = select_place_motion(
        trailing_storage.input);
    TEST_CHECK(after_storage.accepted);
    TEST_CHECK(after_storage.candidate.selection_id ==
               baseline.candidate.selection_id);
    TEST_CHECK(after_storage.candidate.source_id ==
               baseline.candidate.source_id);

    PlaceFixture second_clip = copy_fixture(fixture);
    append_unrelated_clip(second_clip.database);
    const PlaceResult after_second_clip = select_place_motion(
        second_clip.input);
    TEST_CHECK(after_second_clip.accepted);
    TEST_CHECK(after_second_clip.candidate.selection_id ==
               baseline.candidate.selection_id);
    TEST_CHECK(after_second_clip.candidate.source_id ==
               baseline.candidate.source_id);

    PlaceFixture extended_clip = copy_fixture(fixture);
    const auto append_last = [](auto& values, size_t count) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        const std::vector<Value> tail(values.end() -
            static_cast<std::ptrdiff_t>(count), values.end());
        values.insert(values.end(), tail.begin(), tail.end());
    };
    const size_t bones = g1_skeleton::BoneCount;
    append_last(extended_clip.database.positions, bones * 3U);
    append_last(extended_clip.database.velocities, bones * 3U);
    append_last(extended_clip.database.rotations, bones * 4U);
    append_last(extended_clip.database.angular_velocities, bones * 3U);
    append_last(extended_clip.database.foot_contacts, 2U);
    append_last(extended_clip.database.hand_contacts, 2U);
    append_last(extended_clip.database.hand_dof, 14U);
    append_last(extended_clip.database.hand_dof_velocities, 14U);
    append_last(extended_clip.database.phases, 1U);
    append_last(extended_clip.database.time_to_contact, 1U);
    append_last(extended_clip.database.object_positions, 3U);
    append_last(extended_clip.database.object_rotations, 4U);
    append_last(extended_clip.database.object_velocities, 3U);
    append_last(extended_clip.database.object_angular_velocities, 3U);
    extended_clip.database.source_frames.push_back(kFrameCount);
    ++extended_clip.database.frame_count;
    ++extended_clip.database.range_stops.front();
    const PlaceResult after_extension = select_place_motion(
        extended_clip.input);
    TEST_CHECK(after_extension.accepted);
    TEST_CHECK(after_extension.candidate.selection_id ==
               baseline.candidate.selection_id);
    TEST_CHECK(after_extension.candidate.source_id ==
               baseline.candidate.source_id);
}

void test_reverse_certification_rejections() {
    const std::vector<std::function<void(PlaceFixture&)>> invalid = {
        [](PlaceFixture& value) {
            value.database.hand_contacts[
                static_cast<size_t>(kContactFrame + 2) * 2U + 1U] = 0U;
        },
        [](PlaceFixture& value) {
            for (int32_t frame = kHoldFrame; frame < kFrameCount; ++frame) {
                Transform hand = database_object_at(value.database, frame);
                hand.position.x += frame % 2 == 0 ? 0.0F : 0.020001F;
                set_database_hand_world(value.database, frame, hand);
            }
        },
        [](PlaceFixture& value) {
            for (int32_t frame = kHoldFrame; frame < kFrameCount; ++frame) {
                Transform hand = database_object_at(value.database, frame);
                hand.rotation = frame % 2 == 0
                    ? quat()
                    : quat_from_angle_axis(
                        kTenDegrees + 0.0000174533F,
                        vec3(0.0F, 1.0F, 0.0F));
                set_database_hand_world(value.database, frame, hand);
            }
        },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.contact_frame = kLiftFrame;
        },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.lift_frame = kHoldFrame;
        },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.hold_frame = kFrameCount;
        },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.entry_frame = kContactFrame;
        },
        [](PlaceFixture& value) {
            value.database.phases[static_cast<size_t>(kContactFrame)] =
                static_cast<uint8_t>(Phase::Lift);
        },
        [](PlaceFixture& value) {
            value.database.phases[static_cast<size_t>(kLiftFrame)] =
                static_cast<uint8_t>(Phase::Contact);
        },
        [](PlaceFixture& value) {
            value.database.phases[static_cast<size_t>(kHoldFrame)] =
                static_cast<uint8_t>(Phase::Lift);
        },
        [](PlaceFixture& value) {
            value.database.phases[
                static_cast<size_t>(kContactFrame - 1)] =
                static_cast<uint8_t>(Phase::Contact);
        },
        [](PlaceFixture& value) {
            value.database.phases[
                static_cast<size_t>(kLiftFrame + 1)] =
                static_cast<uint8_t>(Phase::Contact);
        },
        [](PlaceFixture& value) {
            value.database.hand_contacts[
                static_cast<size_t>(kHoldFrame + 1) * 2U + 1U] = 2U;
        },
        [](PlaceFixture& value) {
            value.database.active_hands[0] =
                static_cast<uint8_t>(Hand::Left);
        },
        [](PlaceFixture& value) {
            value.database.positions[vec_offset(kContactFrame, kRoot)] =
                std::numeric_limits<float>::quiet_NaN();
        },
        [](PlaceFixture& value) {
            write_bone_rotation(
                value.database,
                kContactFrame,
                kRoot,
                quat(0.0F, 0.0F, 0.0F, 0.0F));
        },
        [](PlaceFixture& value) {
            value.database.object_positions[
                static_cast<size_t>(kLiftFrame) * 3U] =
                std::numeric_limits<float>::infinity();
        },
        [](PlaceFixture& value) {
            write_quat(
                value.database.object_rotations,
                static_cast<size_t>(kLiftFrame),
                quat(2.0F, 0.0F, 0.0F, 0.0F));
        },
        [](PlaceFixture& value) {
            value.database.fps_numerator = 60U;
        },
        [](PlaceFixture& value) {
            value.database.range_starts[0] = 1;
        },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.clip = 1;
        },
    };
    for (const auto& mutate : invalid) {
        PlaceFixture fixture = make_fixture(false);
        mutate(fixture);
        expect_rejected(std::move(fixture));
    }
}

void set_clip_grasp(RecordedPlaceClip& clip, Transform hand_in_object) {
    clip.hand_in_object = hand_in_object;
    for (int32_t frame = clip.entry_frame; frame <= clip.release_frame; ++frame) {
        const Transform hand = compose(
            clip.object_poses.at(static_cast<size_t>(frame)),
            hand_in_object);
        Pose& pose = clip.poses.at(static_cast<size_t>(frame));
        const Transform root = root_transform(pose);
        pose.positions[kRightHand] = quat_mul_vec3(
            quat_inv(root.rotation), hand.position - root.position);
        pose.rotations[kRightHand] = quat_normalize(quat_mul(
            quat_inv(root.rotation), hand.rotation));
    }
}

void set_clip_object_and_hand(
    RecordedPlaceClip& clip,
    int32_t frame,
    Transform object) {
    clip.object_poses.at(static_cast<size_t>(frame)) = object;
    const Transform hand = compose(object, clip.hand_in_object);
    Pose& pose = clip.poses.at(static_cast<size_t>(frame));
    const Transform root = root_transform(pose);
    pose.positions[kRightHand] = quat_mul_vec3(
        quat_inv(root.rotation), hand.position - root.position);
    pose.rotations[kRightHand] = quat_normalize(quat_mul(
        quat_inv(root.rotation), hand.rotation));
}

void test_recorded_rows_validate_locally_before_unique_priority() {
    PlaceFixture malformed_duplicate = make_fixture(false);
    RecordedPlaceClip malformed = make_recorded_clip(101U);
    malformed.entry_frame = malformed.commit_frame;
    malformed_duplicate.library.recorded = {
        malformed, make_recorded_clip(101U)};
    refresh_pointers(malformed_duplicate);
    const PlaceResult surviving =
        select_place_motion(malformed_duplicate.input);
    TEST_CHECK(surviving.accepted);
    TEST_CHECK(surviving.candidate.mode == PlaceMotionMode::RecordedPlace);
    TEST_CHECK(surviving.candidate.source_id == 101U);

    PlaceFixture duplicate_group = make_fixture(false);
    duplicate_group.library.recorded = {
        make_recorded_clip(101U),
        make_recorded_clip(101U),
        make_recorded_clip(103U),
    };
    refresh_pointers(duplicate_group);
    const PlaceResult unrelated = select_place_motion(duplicate_group.input);
    TEST_CHECK(unrelated.accepted);
    TEST_CHECK(unrelated.candidate.mode == PlaceMotionMode::RecordedPlace);
    TEST_CHECK(unrelated.candidate.source_id == 103U);

    PlaceFixture duplicates_only = make_fixture(false);
    duplicates_only.library.recorded = {
        make_recorded_clip(101U), make_recorded_clip(101U)};
    refresh_pointers(duplicates_only);
    const PlaceResult duplicate_fallback =
        select_place_motion(duplicates_only.input);
    TEST_CHECK(duplicate_fallback.accepted);
    TEST_CHECK(duplicate_fallback.candidate.mode ==
               PlaceMotionMode::ReversedPickup);

    PlaceFixture zero_and_unique = make_fixture(false);
    zero_and_unique.library.recorded = {
        make_recorded_clip(0U), make_recorded_clip(104U)};
    refresh_pointers(zero_and_unique);
    const PlaceResult unique = select_place_motion(zero_and_unique.input);
    TEST_CHECK(unique.accepted);
    TEST_CHECK(unique.candidate.mode == PlaceMotionMode::RecordedPlace);
    TEST_CHECK(unique.candidate.source_id == 104U);

    PlaceFixture zero_only = make_fixture(false);
    zero_only.library.recorded = {make_recorded_clip(0U)};
    refresh_pointers(zero_only);
    expect_reversed_fallback(std::move(zero_only));

    PlaceFixture deterministic = make_fixture(false);
    deterministic.library.recorded = {
        make_recorded_clip(102U), make_recorded_clip(101U)};
    refresh_pointers(deterministic);
    const PlaceResult ordered = select_place_motion(deterministic.input);
    TEST_CHECK(ordered.accepted);
    TEST_CHECK(ordered.candidate.source_id == 101U);
    std::reverse(
        deterministic.library.recorded.begin(),
        deterministic.library.recorded.end());
    const PlaceResult reordered = select_place_motion(deterministic.input);
    TEST_CHECK(reordered.accepted);
    TEST_CHECK(reordered.candidate.source_id == 101U);
}

void test_recorded_ranking_uses_aligned_entry_pose_continuity() {
    PlaceFixture fixture = make_fixture(false);
    RecordedPlaceClip discontinuous = make_recorded_clip(101U);
    discontinuous.poses[static_cast<size_t>(discontinuous.entry_frame)]
        .positions[g1_skeleton::LeftToe]
        .x += 0.25F;
    RecordedPlaceClip continuous = make_recorded_clip(202U);
    fixture.library.recorded = {discontinuous, continuous};
    refresh_pointers(fixture);

    const PlaceResult selected = select_place_motion(fixture.input);
    TEST_CHECK(selected.accepted);
    TEST_CHECK(selected.candidate.mode == PlaceMotionMode::RecordedPlace);
    TEST_CHECK(selected.candidate.source_id == continuous.id);

    PlaceFixture far = copy_fixture(fixture);
    const PlaceStagingPreview preview = preview_place_motion(far.input);
    TEST_CHECK(preview.accepted);
    set_current_root_offset(
        far.input,
        preview.staging_root_world,
        far.input.match.maximum_entry_root_error_m + 1.0F,
        0.0F);
    const PlaceResult selected_while_far = select_place_motion(far.input);
    TEST_CHECK(selected_while_far.accepted);
    TEST_CHECK(selected_while_far.candidate.source_id == continuous.id);
}

void test_recorded_structural_physics_and_compatibility_rejections() {
    const std::vector<std::function<void(RecordedPlaceClip&)>> invalid = {
        [](RecordedPlaceClip& clip) { clip.fps_numerator = 60U; },
        [](RecordedPlaceClip& clip) { clip.fps_denominator = 0U; },
        [](RecordedPlaceClip& clip) { clip.poses.pop_back(); },
        [](RecordedPlaceClip& clip) { clip.object_poses.pop_back(); },
        [](RecordedPlaceClip& clip) { clip.active_hand_contacts.pop_back(); },
        [](RecordedPlaceClip& clip) { clip.entry_frame = -1; },
        [](RecordedPlaceClip& clip) { clip.commit_frame = clip.entry_frame; },
        [](RecordedPlaceClip& clip) { clip.release_frame = clip.commit_frame; },
        [](RecordedPlaceClip& clip) {
            clip.retract_stop_frame = clip.release_frame;
        },
        [](RecordedPlaceClip& clip) { clip.retract_stop_frame = 16; },
        [](RecordedPlaceClip& clip) {
            clip.poses[2].positions[0].x =
                std::numeric_limits<float>::quiet_NaN();
        },
        [](RecordedPlaceClip& clip) {
            clip.poses[2].rotations[0] = quat(0.0F, 0.0F, 0.0F, 0.0F);
        },
        [](RecordedPlaceClip& clip) {
            clip.object_poses[2].position.x =
                std::numeric_limits<float>::infinity();
        },
        [](RecordedPlaceClip& clip) {
            clip.object_poses[2].rotation = quat(2.0F, 0.0F, 0.0F, 0.0F);
        },
        [](RecordedPlaceClip& clip) { clip.active_hand_contacts[2] = 2U; },
        [](RecordedPlaceClip& clip) {
            clip.active_hand_contacts[
                static_cast<size_t>(clip.commit_frame)] = 0U;
        },
        [](RecordedPlaceClip& clip) {
            clip.active_hand_contacts[
                static_cast<size_t>(clip.release_frame + 1)] = 1U;
        },
        [](RecordedPlaceClip& clip) {
            clip.poses.push_back(clip.poses.back());
            clip.object_poses.push_back(clip.object_poses.back());
            clip.active_hand_contacts.push_back(1U);
        },
        [](RecordedPlaceClip& clip) { clip.object_profile_id = 0U; },
        [](RecordedPlaceClip& clip) {
            clip.object_profile_id = kProfileId + 1U;
        },
        [](RecordedPlaceClip& clip) {
            clip.object_bounds.center_object.x += 0.001001F;
        },
        [](RecordedPlaceClip& clip) {
            clip.object_bounds.half_extents_object.y += 0.001001F;
        },
        [](RecordedPlaceClip& clip) {
            clip.object_bounds.half_extents_object.x = 0.0F;
        },
        [](RecordedPlaceClip& clip) { clip.hand = Hand::Left; },
        [](RecordedPlaceClip& clip) {
            clip.hand_in_object.position.x = 0.020001F;
        },
        [](RecordedPlaceClip& clip) {
            clip.hand_in_object.rotation = quat_from_angle_axis(
                kTenDegrees + 0.0000174533F,
                vec3(0.0F, 1.0F, 0.0F));
        },
        [](RecordedPlaceClip& clip) {
            clip.source_surface.handle.id = 0U;
        },
        [](RecordedPlaceClip& clip) { clip.source_affordance_id = 999U; },
        [](RecordedPlaceClip& clip) {
            Transform object = clip.object_poses.at(
                static_cast<size_t>(clip.release_frame));
            object.position.x = 0.50F;
            set_clip_object_and_hand(clip, clip.release_frame, object);
        },
        [](RecordedPlaceClip& clip) {
            clip.poses[5].positions[kRightHand].x += 0.020001F;
        },
        [](RecordedPlaceClip& clip) {
            clip.poses[5].rotations[kRightHand] = quat_from_angle_axis(
                kTenDegrees + 0.0000174533F,
                vec3(0.0F, 1.0F, 0.0F));
        },
    };

    for (const auto& mutate : invalid) {
        PlaceFixture fixture = make_fixture(false);
        fixture.library.recorded.push_back(make_recorded_clip(101U));
        mutate(fixture.library.recorded.front());
        refresh_pointers(fixture);
        const PlaceResult fallback = select_place_motion(fixture.input);
        TEST_CHECK(fallback.accepted);
        TEST_CHECK(fallback.candidate.mode ==
                   PlaceMotionMode::ReversedPickup);

        PlaceFixture beside = copy_fixture(fixture);
        beside.library.recorded.push_back(make_recorded_clip(105U));
        refresh_pointers(beside);
        const PlaceResult valid_wins = select_place_motion(beside.input);
        TEST_CHECK(valid_wins.accepted);
        TEST_CHECK(valid_wins.candidate.mode ==
                   PlaceMotionMode::RecordedPlace);
        TEST_CHECK(valid_wins.candidate.source_id == 105U);
    }

    PlaceFixture exact_bounds = make_fixture();
    exact_bounds.library.recorded.front().object_bounds.center_object.x +=
        0.001F;
    exact_bounds.library.recorded.front().object_bounds.half_extents_object.z +=
        0.001F;
    refresh_pointers(exact_bounds);
    TEST_CHECK(select_place_motion(exact_bounds.input).accepted);

    PlaceFixture exact_grasp_position = make_fixture();
    set_clip_grasp(
        exact_grasp_position.library.recorded.front(),
        Transform{vec3(0.020F, 0.0F, 0.0F), quat()});
    refresh_pointers(exact_grasp_position);
    TEST_CHECK(select_place_motion(exact_grasp_position.input).accepted);

    PlaceFixture exact_grasp_rotation = make_fixture();
    set_clip_grasp(
        exact_grasp_rotation.library.recorded.front(),
        Transform{
            vec3(),
            quat_from_angle_axis(
                kTenDegrees, vec3(0.0F, 1.0F, 0.0F))});
    refresh_pointers(exact_grasp_rotation);
    TEST_CHECK(select_place_motion(exact_grasp_rotation.input).accepted);

    PlaceFixture exact_attached_position = make_fixture();
    exact_attached_position.library.recorded.front()
        .poses[5]
        .positions[kRightHand]
        .x += 0.020F;
    refresh_pointers(exact_attached_position);
    TEST_CHECK(select_place_motion(exact_attached_position.input).accepted);

    PlaceFixture exact_attached_rotation = make_fixture();
    exact_attached_rotation.library.recorded.front()
        .poses[5]
        .rotations[kRightHand] = quat_from_angle_axis(
            kTenDegrees, vec3(0.0F, 1.0F, 0.0F));
    refresh_pointers(exact_attached_rotation);
    TEST_CHECK(select_place_motion(exact_attached_rotation.input).accepted);
}

void test_recorded_release_support_boundaries() {
    PlaceFixture footprint = make_fixture();
    footprint.library.recorded.front().source_surface.half_extent_x_m = 0.05F;
    refresh_pointers(footprint);
    TEST_CHECK(select_place_motion(footprint.input).candidate.mode ==
               PlaceMotionMode::RecordedPlace);
    PlaceFixture footprint_over = copy_fixture(footprint);
    Transform object = footprint_over.library.recorded.front().object_poses[12];
    object.position.x += 0.000002F;
    set_clip_object_and_hand(
        footprint_over.library.recorded.front(), 12, object);
    refresh_pointers(footprint_over);
    TEST_CHECK(select_place_motion(footprint_over.input).candidate.mode ==
               PlaceMotionMode::ReversedPickup);

    PlaceFixture maximum_gap = make_fixture();
    object = maximum_gap.library.recorded.front().object_poses[12];
    object.position.y = 0.82F;
    set_clip_object_and_hand(maximum_gap.library.recorded.front(), 12, object);
    refresh_pointers(maximum_gap);
    TEST_CHECK(select_place_motion(maximum_gap.input).candidate.mode ==
               PlaceMotionMode::RecordedPlace);
    PlaceFixture gap_over = copy_fixture(maximum_gap);
    object = gap_over.library.recorded.front().object_poses[12];
    object.position.y += 0.000002F;
    set_clip_object_and_hand(gap_over.library.recorded.front(), 12, object);
    refresh_pointers(gap_over);
    TEST_CHECK(select_place_motion(gap_over.input).candidate.mode ==
               PlaceMotionMode::ReversedPickup);

    PlaceFixture minimum_gap = make_fixture();
    object = minimum_gap.library.recorded.front().object_poses[12];
    object.position.y = 0.795F;
    set_clip_object_and_hand(minimum_gap.library.recorded.front(), 12, object);
    refresh_pointers(minimum_gap);
    TEST_CHECK(select_place_motion(minimum_gap.input).candidate.mode ==
               PlaceMotionMode::RecordedPlace);
    PlaceFixture gap_below = copy_fixture(minimum_gap);
    object = gap_below.library.recorded.front().object_poses[12];
    object.position.y -= 0.000002F;
    set_clip_object_and_hand(gap_below.library.recorded.front(), 12, object);
    refresh_pointers(gap_below);
    TEST_CHECK(select_place_motion(gap_below.input).candidate.mode ==
               PlaceMotionMode::ReversedPickup);

    PlaceFixture overhead = make_fixture();
    overhead.library.recorded.front().source_surface.overhead_clearance_m =
        0.20F;
    refresh_pointers(overhead);
    TEST_CHECK(select_place_motion(overhead.input).candidate.mode ==
               PlaceMotionMode::RecordedPlace);
    PlaceFixture overhead_over = copy_fixture(overhead);
    overhead_over.library.recorded.front().source_surface.overhead_clearance_m =
        0.199998F;
    refresh_pointers(overhead_over);
    TEST_CHECK(select_place_motion(overhead_over.input).candidate.mode ==
               PlaceMotionMode::ReversedPickup);
}

void test_selection_certifies_mapped_support_volume_clearance() {
    PlaceFixture touching = make_fixture();
    RecordedPlaceClip& touching_clip = touching.library.recorded.front();
    Transform object = touching_clip.object_poses[5];
    object.position.y = 0.80F;
    set_clip_object_and_hand(touching_clip, 5, object);
    refresh_pointers(touching);
    const PlaceResult touching_result = select_place_motion(touching.input);
    TEST_CHECK(touching_result.accepted);
    TEST_CHECK(touching_result.candidate.mode ==
               PlaceMotionMode::ReversedPickup);

    PlaceFixture clear = make_fixture();
    RecordedPlaceClip& clear_clip = clear.library.recorded.front();
    object = clear_clip.object_poses[5];
    object.position.y = std::nextafter(
        0.80F, std::numeric_limits<float>::infinity());
    set_clip_object_and_hand(clear_clip, 5, object);
    refresh_pointers(clear);
    const PlaceResult clear_result = select_place_motion(clear.input);
    TEST_CHECK(clear_result.accepted);
    TEST_CHECK(clear_result.candidate.mode == PlaceMotionMode::RecordedPlace);

    PlaceFixture swept = make_fixture();
    RecordedPlaceClip& swept_clip = swept.library.recorded.front();
    object = swept_clip.object_poses[5];
    object.position = vec3(-1.0F, 0.50F, 0.0F);
    set_clip_object_and_hand(swept_clip, 5, object);
    object.position.x = 1.0F;
    set_clip_object_and_hand(swept_clip, 6, object);
    refresh_pointers(swept);
    const PlaceResult swept_result = select_place_motion(swept.input);
    TEST_CHECK(swept_result.accepted);
    TEST_CHECK(swept_result.candidate.mode ==
               PlaceMotionMode::ReversedPickup);

    PlaceFixture seam = make_fixture();
    RecordedPlaceClip& seam_clip = seam.library.recorded.front();
    for (int32_t frame = seam_clip.entry_frame;
         frame < seam_clip.release_frame;
         ++frame) {
        object = seam_clip.object_poses[static_cast<size_t>(frame)];
        object.position = vec3(1.0F, 0.50F, 0.0F);
        set_clip_object_and_hand(seam_clip, frame, object);
    }
    const Transform current_root = root_transform(seam.input.current_pose);
    seam.input.current_object_world = {
        seam.input.surface.support_volume_world.position +
            vec3(-1.0F, 0.15F, 0.0F),
        quat(),
    };
    seam.input.current_pose.positions[kRightHand] = quat_mul_vec3(
        quat_inv(current_root.rotation),
        seam.input.current_object_world.position - current_root.position);
    seam.input.current_pose.rotations[kRightHand] = quat_normalize(quat_mul(
        quat_inv(current_root.rotation),
        seam.input.current_object_world.rotation));
    refresh_pointers(seam);
    const PlaceResult seam_result = select_place_motion(seam.input);
    TEST_CHECK(!seam_result.accepted);
    TEST_CHECK(seam_result.reason == Reason::BlockedPath);

    PlaceFixture reverse_blocked = make_fixture(false);
    Transform blocked_hand = hand_transform(
        pose_at_frame(reverse_blocked.database, kLiftFrame));
    blocked_hand.position.y = 0.50F;
    set_database_hand_world(
        reverse_blocked.database, kLiftFrame, blocked_hand);
    const PlaceResult reverse_result = select_place_motion(
        reverse_blocked.input);
    TEST_CHECK(!reverse_result.accepted);
    TEST_CHECK(reverse_result.reason == Reason::BlockedPath);

    PlaceFixture poisoned_object = make_fixture(false);
    object = database_object_at(poisoned_object.database, kLiftFrame);
    object.position.y = 0.50F;
    write_vec(
        poisoned_object.database.object_positions,
        static_cast<size_t>(kLiftFrame),
        object.position);
    const PlaceResult poison_result = select_place_motion(
        poisoned_object.input);
    TEST_CHECK(poison_result.accepted);
    TEST_CHECK(poison_result.candidate.mode ==
               PlaceMotionMode::ReversedPickup);
}

void test_selection_runs_real_ik_with_exact_config() {
    PlaceFixture recorded = make_fixture();
    set_clip_grasp(
        recorded.library.recorded.front(),
        Transform{vec3(0.020F, 0.0F, 0.0F), quat()});
    recorded.input.ik.accepted_position_m = 0.0F;
    recorded.input.ik.maximum_iterations = 0;
    refresh_pointers(recorded);
    const PlaceResult recorded_result = select_place_motion(recorded.input);
    TEST_CHECK(recorded_result.accepted);
    TEST_CHECK(recorded_result.candidate.mode ==
               PlaceMotionMode::ReversedPickup);

    PlaceFixture reverse = make_fixture(false);
    for (int32_t frame = 0; frame < kFrameCount; ++frame) {
        Transform shifted = database_object_at(reverse.database, frame);
        shifted.position.y += 0.005F;
        set_database_object_and_hand(reverse.database, frame, shifted);
    }
    reverse.input.ik.maximum_request_position_m = 0.005F;
    reverse.input.ik.accepted_position_m = 0.0F;
    reverse.input.ik.maximum_iterations = 0;
    const PlaceResult reverse_result = select_place_motion(reverse.input);
    TEST_CHECK(!reverse_result.accepted);
    TEST_CHECK(reverse_result.reason == Reason::CorrectionLimit);

    reverse.input.ik.accepted_position_m = 0.006F;
    const PlaceResult relaxed_result = select_place_motion(reverse.input);
    TEST_CHECK(relaxed_result.accepted);
    TEST_CHECK(relaxed_result.candidate.mode ==
               PlaceMotionMode::ReversedPickup);
}

void test_placement_fit_failure_reasons_are_preserved() {
    PlaceFixture solved_release = make_fixture();
    set_clip_grasp(
        solved_release.library.recorded.front(),
        Transform{vec3(0.020F, 0.0F, 0.0F), quat()});
    solved_release.input.place_affordance.object_in_surface.position.x =
        solved_release.input.surface.half_extent_x_m -
        solved_release.input.held_object_bounds.half_extents_object.x -
        solved_release.input.place_affordance.clearance_radius;
    solved_release.input.surface.affordances.front() =
        solved_release.input.place_affordance;
    solved_release.input.ik.accepted_position_m = 0.021F;
    solved_release.input.ik.maximum_iterations = 0;
    solved_release.input.pickup_candidate.clip = -1;
    refresh_pointers(solved_release);
    const PlaceResult solved_release_result = select_place_motion(
        solved_release.input);
    TEST_CHECK(!solved_release_result.accepted);
    TEST_CHECK(solved_release_result.reason ==
               Reason::PlacementOutOfBounds);

    PlaceFixture source_release = make_fixture();
    source_release.library.recorded.front()
        .source_surface.half_extent_x_m = 0.049F;
    source_release.input.pickup_candidate.clip = -1;
    refresh_pointers(source_release);
    const PlaceResult source_release_result = select_place_motion(
        source_release.input);
    TEST_CHECK(!source_release_result.accepted);
    TEST_CHECK(source_release_result.reason ==
               Reason::PlacementOutOfBounds);
}

uint64_t accepted_selection_id(PlaceFixture& fixture) {
    refresh_pointers(fixture);
    const PlaceResult result = select_place_motion(fixture.input);
    TEST_CHECK(result.accepted);
    TEST_CHECK(result.candidate.selection_id != 0U);
    return result.candidate.selection_id;
}

void assert_selection_id_changes(
    const PlaceFixture& baseline,
    const std::function<void(PlaceFixture&)>& mutate) {
    static int identity_case = 0;
    const int case_number = ++identity_case;
    PlaceFixture original = copy_fixture(baseline);
    const uint64_t before = accepted_selection_id(original);
    PlaceFixture changed = copy_fixture(baseline);
    mutate(changed);
    const uint64_t after = accepted_selection_id(changed);
    if (after == 0U || after == before) {
        std::cerr << "identity mutation case " << case_number
                  << " before " << before << " after " << after << '\n';
    }
    TEST_CHECK(after != before);
}

void test_complete_snapshot_identity_and_pointer_independence() {
    PlaceFixture base = make_fixture();
    base.library.recorded.push_back(make_recorded_clip(202U));
    refresh_pointers(base);
    const uint64_t baseline = accepted_selection_id(base);

    PlaceFixture copied = copy_fixture(base);
    TEST_CHECK(copied.input.pickup_database != base.input.pickup_database);
    TEST_CHECK(copied.input.library != base.input.library);
    TEST_CHECK(accepted_selection_id(copied) == baseline);

    const std::vector<std::function<void(PlaceFixture&)>> current_pose = {
        [](PlaceFixture& value) {
            value.input.current_pose.positions[g1_skeleton::LeftToe].x +=
                0.001F;
        },
        [](PlaceFixture& value) {
            value.input.current_pose.velocities[g1_skeleton::LeftToe].y +=
                0.001F;
        },
        [](PlaceFixture& value) {
            value.input.current_pose.rotations[g1_skeleton::LeftToe] =
                quat_from_angle_axis(
                    0.001F, vec3(1.0F, 0.0F, 0.0F));
        },
        [](PlaceFixture& value) {
            value.input.current_pose
                .angular_velocities[g1_skeleton::LeftToe]
                .z += 0.001F;
        },
        [](PlaceFixture& value) { value.input.current_pose.hand_dof[3] += 0.001F; },
        [](PlaceFixture& value) {
            value.input.current_pose.hand_dof_velocities[4] += 0.001F;
        },
        [](PlaceFixture& value) {
            value.input.current_pose.foot_contacts[0] ^= 1U;
        },
    };
    for (const auto& mutate : current_pose) {
        assert_selection_id_changes(base, mutate);
    }

    const std::vector<std::function<void(PlaceFixture&)>> held_snapshot = {
        [](PlaceFixture& value) {
            value.input.current_object_world.position.x += 0.001F;
        },
        [](PlaceFixture& value) {
            value.input.current_object_world.rotation = quat_from_angle_axis(
                0.001F, vec3(0.0F, 1.0F, 0.0F));
        },
        [](PlaceFixture& value) { ++value.input.held_target.id; },
        [](PlaceFixture& value) { ++value.input.held_target.generation; },
        [](PlaceFixture& value) { ++value.input.held_object_profile_id; },
        [](PlaceFixture& value) {
            value.input.held_object_bounds.center_object.x += 0.0001F;
        },
        [](PlaceFixture& value) {
            value.input.held_object_bounds.half_extents_object.z += 0.0001F;
        },
        [](PlaceFixture& value) { ++value.input.held_affordance.id; },
        [](PlaceFixture& value) {
            value.input.held_affordance.hand_in_object.position.x += 0.001F;
        },
        [](PlaceFixture& value) {
            value.input.held_affordance.hand_in_object.rotation =
                quat_from_angle_axis(
                    0.001F, vec3(0.0F, 1.0F, 0.0F));
        },
        [](PlaceFixture& value) {
            value.input.held_affordance.approach_direction_object =
                normalize(vec3(0.001F, 1.0F, 0.0F));
        },
        [](PlaceFixture& value) {
            value.input.held_affordance.clearance_radius += 0.001F;
        },
        [](PlaceFixture& value) { value.input.object_dimensions.x += 0.001F; },
        [](PlaceFixture& value) { value.input.object_dimensions.y += 0.001F; },
        [](PlaceFixture& value) { value.input.object_dimensions.z += 0.001F; },
    };
    for (const auto& mutate : held_snapshot) {
        assert_selection_id_changes(base, mutate);
    }

    const std::vector<std::function<void(PlaceFixture&)>> surface_snapshot = {
        [](PlaceFixture& value) { ++value.input.surface.handle.id; },
        [](PlaceFixture& value) { ++value.input.surface.handle.generation; },
        [](PlaceFixture& value) {
            value.input.surface.surface_world.position.x += 0.001F;
            value.input.surface.support_volume_world.position.x += 0.001F;
        },
        [](PlaceFixture& value) {
            const quat yaw = quat_from_angle_axis(
                0.001F, vec3(0.0F, 1.0F, 0.0F));
            value.input.surface.surface_world.rotation = yaw;
            value.input.surface.support_volume_world.rotation = yaw;
        },
        [](PlaceFixture& value) {
            value.input.surface.support_volume_world.position.x += 0.0001F;
        },
        [](PlaceFixture& value) {
            value.input.surface.support_volume_world.rotation =
                quat_from_angle_axis(
                    0.0001F, vec3(0.0F, 1.0F, 0.0F));
        },
        [](PlaceFixture& value) {
            value.input.surface.support_volume_size.x += 0.001F;
        },
        [](PlaceFixture& value) {
            value.input.surface.half_extent_x_m += 0.001F;
        },
        [](PlaceFixture& value) {
            value.input.surface.half_extent_z_m += 0.001F;
        },
        [](PlaceFixture& value) {
            value.input.surface.overhead_clearance_m += 0.001F;
        },
        [](PlaceFixture& value) { ++value.input.surface.affordances[1].id; },
        [](PlaceFixture& value) {
            value.input.surface.affordances[1].object_in_surface.position.x +=
                0.001F;
        },
        [](PlaceFixture& value) {
            value.input.surface.affordances[1].object_in_surface.rotation =
                quat_from_angle_axis(
                    0.001F, vec3(0.0F, 1.0F, 0.0F));
        },
        [](PlaceFixture& value) {
            value.input.surface.affordances[1].support_point_object.x +=
                0.001F;
        },
        [](PlaceFixture& value) {
            value.input.surface.affordances[1].approach_direction_surface =
                normalize(vec3(0.001F, 1.0F, 0.0F));
        },
        [](PlaceFixture& value) {
            value.input.surface.affordances[1].clearance_radius += 0.001F;
        },
    };
    for (const auto& mutate : surface_snapshot) {
        assert_selection_id_changes(base, mutate);
    }

    const std::vector<std::function<void(PlaceFixture&)>> requested_affordance = {
        [](PlaceFixture& value) {
            value.input.place_affordance.id += 100U;
            value.input.surface.affordances[0].id =
                value.input.place_affordance.id;
        },
        [](PlaceFixture& value) {
            value.input.place_affordance.object_in_surface.position.x += 0.001F;
            value.input.surface.affordances[0] = value.input.place_affordance;
        },
        [](PlaceFixture& value) {
            value.input.place_affordance.object_in_surface.rotation =
                quat_from_angle_axis(
                    0.001F, vec3(0.0F, 1.0F, 0.0F));
            value.input.surface.affordances[0] = value.input.place_affordance;
        },
        [](PlaceFixture& value) {
            value.input.place_affordance.support_point_object.x += 0.001F;
            value.input.surface.affordances[0] = value.input.place_affordance;
        },
        [](PlaceFixture& value) {
            value.input.place_affordance.approach_direction_surface =
                normalize(vec3(0.001F, 1.0F, 0.0F));
            value.input.surface.affordances[0] = value.input.place_affordance;
        },
        [](PlaceFixture& value) {
            value.input.place_affordance.clearance_radius += 0.001F;
            value.input.surface.affordances[0] = value.input.place_affordance;
        },
    };
    for (const auto& mutate : requested_affordance) {
        assert_selection_id_changes(base, mutate);
    }

    const std::vector<std::function<void(PlaceFixture&)>> configs = {
        [](PlaceFixture& value) { value.input.timing.playback_speed = 0.99F; },
        [](PlaceFixture& value) {
            value.input.timing.entry_blend_seconds = 0.24F;
        },
        [](PlaceFixture& value) {
            value.input.timing.reversed_commit_seconds = 0.49F;
        },
        [](PlaceFixture& value) {
            value.input.timing.maximum_alignment_seconds = 0.99F;
        },
        [](PlaceFixture& value) {
            value.input.match.maximum_entry_root_error_m = 0.24F;
        },
        [](PlaceFixture& value) {
            value.input.match.maximum_entry_yaw_error_radians = 0.43F;
        },
    };
    for (const auto& mutate : configs) {
        assert_selection_id_changes(base, mutate);
    }

    const std::vector<std::function<void(PlaceFixture&)>> pickup_candidate = {
        [](PlaceFixture& value) { value.input.pickup_candidate.entry_frame = 1; },
        [](PlaceFixture& value) { value.input.pickup_candidate.contact_frame = 9; },
        [](PlaceFixture& value) { value.input.pickup_candidate.lift_frame = 13; },
        [](PlaceFixture& value) { value.input.pickup_candidate.hold_frame = 19; },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.scene_from_source.position.x += 0.001F;
        },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.scene_from_source.rotation =
                quat_from_angle_axis(
                    0.001F, vec3(0.0F, 1.0F, 0.0F));
        },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.entry_root_offset.x += 0.001F;
        },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.entry_yaw_offset += 0.001F;
        },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.total_cost += 0.001F;
        },
        [](PlaceFixture& value) {
            value.input.pickup_candidate.group_costs[3] += 0.001F;
        },
    };
    for (const auto& mutate : pickup_candidate) {
        assert_selection_id_changes(base, mutate);
    }

    const std::vector<std::function<void(PlaceFixture&)>> pickup_source = {
        [](PlaceFixture& value) {
            value.database.positions[vec_offset(2, g1_skeleton::LeftToe)] +=
                0.001F;
        },
        [](PlaceFixture& value) {
            value.database.velocities[vec_offset(2, g1_skeleton::LeftToe) + 1U] +=
                0.001F;
        },
        [](PlaceFixture& value) {
            write_bone_rotation(
                value.database,
                2,
                g1_skeleton::LeftToe,
                quat_from_angle_axis(
                    0.001F, vec3(1.0F, 0.0F, 0.0F)));
        },
        [](PlaceFixture& value) {
            value.database.angular_velocities[
                vec_offset(2, g1_skeleton::LeftToe) + 2U] += 0.001F;
        },
        [](PlaceFixture& value) { value.database.foot_contacts[4] ^= 1U; },
        [](PlaceFixture& value) { value.database.hand_dof[4] += 0.001F; },
        [](PlaceFixture& value) {
            value.database.hand_dof_velocities[5] += 0.001F;
        },
        [](PlaceFixture& value) {
            value.database.time_to_contact[2] += 0.001F;
        },
        [](PlaceFixture& value) {
            value.database.object_velocities[2 * 3U] += 0.001F;
        },
        [](PlaceFixture& value) {
            value.database.object_angular_velocities[2 * 3U + 1U] += 0.001F;
        },
        [](PlaceFixture& value) { value.database.table_positions[0] += 0.001F; },
        [](PlaceFixture& value) { value.database.table_sizes[0] += 0.001F; },
        [](PlaceFixture& value) {
            value.database.object_dimensions[0] += 0.001F;
        },
        [](PlaceFixture& value) {
            value.database.grasp_positions_object[0] += 0.001F;
        },
        [](PlaceFixture& value) {
            value.database.approach_directions_object[0] += 0.001F;
        },
        [](PlaceFixture& value) { ++value.database.source_frames[2]; },
    };
    for (const auto& mutate : pickup_source) {
        assert_selection_id_changes(base, mutate);
    }

    const std::vector<std::function<void(PlaceFixture&)>> consulted_row = {
        [](PlaceFixture& value) {
            value.library.recorded[1].poses[2]
                .positions[g1_skeleton::LeftToe]
                .x += 0.001F;
        },
        [](PlaceFixture& value) {
            value.library.recorded[1].poses[2]
                .velocities[g1_skeleton::LeftToe]
                .x += 0.001F;
        },
        [](PlaceFixture& value) {
            value.library.recorded[1].poses[2]
                .rotations[g1_skeleton::LeftToe] = quat_from_angle_axis(
                    0.001F, vec3(1.0F, 0.0F, 0.0F));
        },
        [](PlaceFixture& value) {
            value.library.recorded[1].poses[2]
                .angular_velocities[g1_skeleton::LeftToe]
                .x += 0.001F;
        },
        [](PlaceFixture& value) {
            value.library.recorded[1].poses[2].hand_dof[0] += 0.001F;
        },
        [](PlaceFixture& value) {
            value.library.recorded[1].poses[2].hand_dof_velocities[0] +=
                0.001F;
        },
        [](PlaceFixture& value) {
            value.library.recorded[1].poses[2].foot_contacts[0] ^= 1U;
        },
        [](PlaceFixture& value) {
            value.library.recorded[1].object_poses[14].position.x += 0.001F;
        },
        [](PlaceFixture& value) {
            value.library.recorded[1].active_hand_contacts[14] = 2U;
        },
        [](PlaceFixture& value) { ++value.library.recorded[1].id; },
        [](PlaceFixture& value) {
            ++value.library.recorded[1].source_surface.handle.generation;
        },
    };
    for (const auto& mutate : consulted_row) {
        assert_selection_id_changes(base, mutate);
    }

    const std::vector<std::function<void(PlaceFixture&)>> selected_events = {
        [](PlaceFixture& value) { value.library.recorded[0].entry_frame = 1; },
        [](PlaceFixture& value) { value.library.recorded[0].commit_frame = 7; },
        [](PlaceFixture& value) {
            value.library.recorded[0].release_frame = 11;
            value.library.recorded[0].active_hand_contacts[12] = 0U;
        },
        [](PlaceFixture& value) {
            value.library.recorded[0].retract_stop_frame = 14;
        },
    };
    for (const auto& mutate : selected_events) {
        assert_selection_id_changes(base, mutate);
    }

    const PlaceStagingPreview preview = preview_place_motion(base.input);
    assert_selection_id_changes(base, [&](PlaceFixture& value) {
        set_current_root_offset(
            value.input, preview.staging_root_world, 0.01F, 0.01F);
    });
}

void test_ik_fingerprint_is_complete_and_configuration_only() {
    PlaceFixture base = make_fixture();
    const PlaceStagingPreview original = preview_place_motion(base.input);
    TEST_CHECK(original.accepted);
    TEST_CHECK(original.ik_config_fingerprint != 0U);
    using IKFloatMember = float IKConfig::*;
    constexpr std::array<IKFloatMember, 8> fields = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (IKFloatMember field : fields) {
        PlaceFixture changed = copy_fixture(base);
        if (field == &IKConfig::maximum_request_position_m ||
            field == &IKConfig::maximum_request_orientation_radians) {
            changed.input.ik.*field *= 0.9F;
        } else {
            changed.input.ik.*field *= 1.1F;
        }
        const PlaceStagingPreview after = preview_place_motion(changed.input);
        TEST_CHECK(after.accepted);
        TEST_CHECK(after.ik_config_fingerprint !=
                   original.ik_config_fingerprint);
        TEST_CHECK(after.candidate.selection_id !=
                   original.candidate.selection_id);
    }
    PlaceFixture iterations = copy_fixture(base);
    ++iterations.input.ik.maximum_iterations;
    const PlaceStagingPreview iteration_preview =
        preview_place_motion(iterations.input);
    TEST_CHECK(iteration_preview.accepted);
    TEST_CHECK(iteration_preview.ik_config_fingerprint !=
               original.ik_config_fingerprint);

    PlaceFixture pose_only = copy_fixture(base);
    pose_only.input.current_pose.positions[g1_skeleton::LeftToe].x += 0.001F;
    pose_only.input.current_object_world.position.x += 0.001F;
    ++pose_only.input.held_target.id;
    const PlaceStagingPreview pose_preview = preview_place_motion(pose_only.input);
    TEST_CHECK(pose_preview.accepted);
    TEST_CHECK(pose_preview.ik_config_fingerprint ==
               original.ik_config_fingerprint);
}

void test_canonical_identity_normalizes_negative_zero_and_quaternion_sign() {
    PlaceFixture zero = make_fixture();
    zero.input.current_pose.positions[g1_skeleton::LeftToe].x = 0.0F;
    zero.input.ik.maximum_request_position_m = 0.0F;
    const PlaceStagingPreview positive_zero = preview_place_motion(zero.input);
    TEST_CHECK(positive_zero.accepted);
    PlaceFixture negative_zero = copy_fixture(zero);
    negative_zero.input.current_pose.positions[g1_skeleton::LeftToe].x = -0.0F;
    negative_zero.input.ik.maximum_request_position_m = -0.0F;
    const PlaceStagingPreview negative_zero_preview =
        preview_place_motion(negative_zero.input);
    TEST_CHECK(negative_zero_preview.accepted);
    TEST_CHECK(negative_zero_preview.candidate.selection_id ==
               positive_zero.candidate.selection_id);
    TEST_CHECK(negative_zero_preview.ik_config_fingerprint ==
               positive_zero.ik_config_fingerprint);

    PlaceFixture positive = make_fixture();
    positive.input.current_pose.rotations[g1_skeleton::LeftToe] =
        quat(0.0F, 1.0F, 0.0F, 0.0F);
    positive.library.recorded[0]
        .poses[2]
        .rotations[g1_skeleton::LeftToe] = quat(0.0F, 0.0F, 1.0F, 0.0F);
    refresh_pointers(positive);
    const uint64_t positive_id = accepted_selection_id(positive);
    PlaceFixture negative = copy_fixture(positive);
    negative.input.current_pose.rotations[g1_skeleton::LeftToe] =
        quat(-0.0F, -1.0F, -0.0F, -0.0F);
    negative.library.recorded[0]
        .poses[2]
        .rotations[g1_skeleton::LeftToe] =
            quat(-0.0F, -0.0F, -1.0F, -0.0F);
    refresh_pointers(negative);
    TEST_CHECK(accepted_selection_id(negative) == positive_id);

    PlaceFixture source_positive = make_fixture();
    write_bone_rotation(
        source_positive.database,
        2,
        g1_skeleton::LeftToe,
        quat(0.0F, 1.0F, 0.0F, 0.0F));
    write_quat(
        source_positive.database.object_rotations,
        2U,
        quat(0.0F, 0.0F, 1.0F, 0.0F));
    write_quat(
        source_positive.database.table_rotations,
        0U,
        quat(0.0F, 0.0F, 0.0F, 1.0F));
    write_quat(
        source_positive.database.grasp_rotations_object,
        0U,
        quat(0.0F, 1.0F, 0.0F, 0.0F));
    refresh_pointers(source_positive);
    const uint64_t source_positive_id =
        accepted_selection_id(source_positive);
    PlaceFixture source_negative = copy_fixture(source_positive);
    write_bone_rotation(
        source_negative.database,
        2,
        g1_skeleton::LeftToe,
        quat(-0.0F, -1.0F, -0.0F, -0.0F));
    write_quat(
        source_negative.database.object_rotations,
        2U,
        quat(-0.0F, -0.0F, -1.0F, -0.0F));
    write_quat(
        source_negative.database.table_rotations,
        0U,
        quat(-0.0F, -0.0F, -0.0F, -1.0F));
    write_quat(
        source_negative.database.grasp_rotations_object,
        0U,
        quat(-0.0F, -1.0F, -0.0F, -0.0F));
    refresh_pointers(source_negative);
    TEST_CHECK(accepted_selection_id(source_negative) == source_positive_id);
}

PlaceResult select_reverse_at_speed(PlaceFixture& fixture, float speed) {
    remove_recorded(fixture);
    fixture.input.timing.playback_speed = speed;
    return select_place_motion(fixture.input);
}

void test_commit_event_derivation_and_runtime_bounds() {
    for (float speed : {0.85F, 1.0F, 1.15F}) {
        PlaceFixture fixture = make_fixture(false);
        const PlaceResult result = select_reverse_at_speed(fixture, speed);
        TEST_CHECK(result.accepted);
        TEST_CHECK(result.candidate.mode ==
                   PlaceMotionMode::ReversedPickup);
        const int32_t offset = static_cast<int32_t>(std::floor(
            fixture.input.timing.reversed_commit_seconds * 25.0F * speed));
        TEST_CHECK(offset > 0);
        TEST_CHECK(result.candidate.commit_frame ==
                   result.candidate.entry_frame - offset);
        TEST_CHECK(result.candidate.commit_frame <
                   result.candidate.entry_frame);
        TEST_CHECK(result.candidate.commit_frame >
                   result.candidate.release_frame);
        const int32_t commit_ticks = static_cast<int32_t>(std::ceil(
            static_cast<double>(offset) / speed));
        const float elapsed = static_cast<float>(commit_ticks) / 25.0F;
        TEST_CHECK(elapsed >= fixture.input.timing.entry_blend_seconds);
        TEST_CHECK(elapsed <= fixture.input.timing.maximum_alignment_seconds);
    }

    for (float speed : {0.85F, 1.0F, 1.15F}) {
        PlaceFixture recorded = make_fixture();
        disable_reverse_tier(recorded);
        recorded.input.timing.playback_speed = speed;
        recorded.input.timing.reversed_commit_seconds = 0.20F;
        const int32_t delta =
            recorded.library.recorded.front().commit_frame -
            recorded.library.recorded.front().entry_frame;
        const int32_t commit_ticks = static_cast<int32_t>(std::ceil(
            static_cast<double>(delta) / speed));
        const float observable_elapsed =
            static_cast<float>(commit_ticks) / 25.0F;
        recorded.input.timing.maximum_alignment_seconds = observable_elapsed;
        const PlaceResult exact = select_place_motion(recorded.input);
        TEST_CHECK(exact.accepted);
        TEST_CHECK(exact.candidate.mode == PlaceMotionMode::RecordedPlace);

        PlaceFixture too_late = copy_fixture(recorded);
        too_late.input.timing.maximum_alignment_seconds = std::nextafter(
            observable_elapsed, 0.0F);
        TEST_CHECK(!select_place_motion(too_late.input).accepted);
    }

    for (float speed : {0.85F, 1.0F, 1.15F}) {
        PlaceFixture reversed = make_fixture(false);
        reversed.input.timing.playback_speed = speed;
        const int32_t offset = static_cast<int32_t>(std::floor(
            reversed.input.timing.reversed_commit_seconds * 25.0F * speed));
        const int32_t commit_ticks = static_cast<int32_t>(std::ceil(
            static_cast<double>(offset) / speed));
        const float observable_elapsed =
            static_cast<float>(commit_ticks) / 25.0F;
        reversed.input.timing.entry_blend_seconds = observable_elapsed;
        const PlaceResult exact = select_place_motion(reversed.input);
        TEST_CHECK(exact.accepted);
        TEST_CHECK(exact.candidate.mode == PlaceMotionMode::ReversedPickup);

        PlaceFixture too_early = copy_fixture(reversed);
        too_early.input.timing.entry_blend_seconds = std::nextafter(
            observable_elapsed, std::numeric_limits<float>::infinity());
        TEST_CHECK(!select_place_motion(too_early.input).accepted);
    }

    PlaceFixture zero_offset = make_fixture(false);
    zero_offset.input.timing.entry_blend_seconds = 0.0001F;
    zero_offset.input.timing.reversed_commit_seconds = 0.001F;
    zero_offset.input.timing.maximum_alignment_seconds = 0.10F;
    expect_rejected(std::move(zero_offset));

    PlaceFixture beyond_release = make_fixture(false);
    beyond_release.input.timing.playback_speed = 1.15F;
    beyond_release.input.timing.reversed_commit_seconds = 0.60F;
    beyond_release.input.timing.maximum_alignment_seconds = 0.80F;
    expect_rejected(std::move(beyond_release));

    PlaceFixture below_entry_blend = make_fixture(false);
    below_entry_blend.input.timing.entry_blend_seconds = 0.49F;
    below_entry_blend.input.timing.reversed_commit_seconds = 0.50F;
    expect_rejected(std::move(below_entry_blend));

    for (float speed : {0.85F, 1.0F, 1.15F}) {
        PlaceFixture recorded = make_fixture();
        recorded.input.timing.playback_speed = speed;
        const PlaceResult result = select_place_motion(recorded.input);
        TEST_CHECK(result.accepted);
        TEST_CHECK(result.candidate.mode == PlaceMotionMode::RecordedPlace);
        TEST_CHECK(result.candidate.commit_frame ==
                   recorded.library.recorded.front().commit_frame);
    }

    PlaceFixture recorded_too_early = make_fixture();
    recorded_too_early.input.timing.playback_speed = 1.15F;
    recorded_too_early.input.timing.entry_blend_seconds = 0.28F;
    const PlaceResult early = select_place_motion(recorded_too_early.input);
    TEST_CHECK(early.accepted);
    TEST_CHECK(early.candidate.mode == PlaceMotionMode::RecordedPlace);

    PlaceFixture recorded_279_limit = make_fixture();
    disable_reverse_tier(recorded_279_limit);
    recorded_279_limit.input.timing.playback_speed = 1.15F;
    recorded_279_limit.input.timing.reversed_commit_seconds = 0.20F;
    recorded_279_limit.input.timing.maximum_alignment_seconds = 0.279F;
    TEST_CHECK(!select_place_motion(recorded_279_limit.input).accepted);

    PlaceFixture recorded_too_late = make_fixture();
    recorded_too_late.input.timing.entry_blend_seconds = 0.20F;
    recorded_too_late.input.timing.reversed_commit_seconds = 0.20F;
    recorded_too_late.input.timing.maximum_alignment_seconds = 0.30F;
    const PlaceResult late = select_place_motion(recorded_too_late.input);
    if (!late.accepted) {
        std::cerr << "late timing rejected reason "
                  << static_cast<int>(late.reason) << '\n';
    }
    TEST_CHECK(late.accepted);
    TEST_CHECK(late.candidate.mode == PlaceMotionMode::ReversedPickup);
}

Pose reachable_arm_pose(Pose pose, float link_length) {
    for (const HingeJoint& joint : kRightArm) {
        pose.positions[static_cast<size_t>(joint.bone)] = vec3();
        pose.rotations[static_cast<size_t>(joint.bone)] =
            joint.rest_rotation;
    }
    pose.positions[static_cast<size_t>(g1_skeleton::RightElbow)] =
        vec3(0.0F, link_length, 0.0F);
    pose.positions[static_cast<size_t>(g1_skeleton::RightWrist)] =
        vec3(0.0F, link_length, 0.0F);
    return pose;
}

Pose reachable_arm_target(Pose pose, float radians) {
    pose.rotations[static_cast<size_t>(g1_skeleton::RightShoulderRoll)] =
        quat_normalize(quat_mul(
            kRightArm[1].rest_rotation,
            quat_from_angle_axis(radians, kRightArm[1].axis)));
    pose.rotations[static_cast<size_t>(g1_skeleton::RightWristRoll)] =
        quat_normalize(quat_mul(
            kRightArm[4].rest_rotation,
            quat_from_angle_axis(-2.0F * radians, kRightArm[4].axis)));
    return pose;
}

void write_database_pose(Database& database, int32_t frame, const Pose& pose) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        write_bone_position(database, frame, bone, pose.positions[bone]);
        write_bone_rotation(database, frame, bone, pose.rotations[bone]);
    }
}

Transform planar_alignment_for_test(Transform source, Transform target) {
    const float yaw = std::atan2(
        std::sin(measured_yaw(target.rotation) -
                 measured_yaw(source.rotation)),
        std::cos(measured_yaw(target.rotation) -
                 measured_yaw(source.rotation)));
    const quat rotation = quat_from_angle_axis(
        yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source = quat_mul_vec3(rotation, source.position);
    return {
        vec3(
            target.position.x - rotated_source.x,
            0.0F,
            target.position.z - rotated_source.z),
        rotation,
    };
}

float configure_reachable_reverse_request(PlaceFixture& fixture) {
    constexpr float angle = 0.40F;
    float link_length = 0.75F;
    Pose contact_source{};
    Pose target_pose{};
    Transform source_hand{};
    Transform target_hand{};
    for (int iteration = 0; iteration < 6; ++iteration) {
        contact_source = reachable_arm_pose(
            pose_at_frame(fixture.database, kContactFrame), link_length);
        target_pose = reachable_arm_target(contact_source, angle);
        source_hand = hand_transform(contact_source);
        target_hand = hand_transform(target_pose);
        const float vertical_error = std::abs(
            source_hand.position.y - target_hand.position.y);
        link_length *= kPositionLimit / vertical_error;
    }
    for (;;) {
        contact_source = reachable_arm_pose(
            pose_at_frame(fixture.database, kContactFrame), link_length);
        target_pose = reachable_arm_target(contact_source, angle);
        source_hand = hand_transform(contact_source);
        target_hand = hand_transform(target_pose);
        if (std::abs(source_hand.position.y - target_hand.position.y) <=
            kPositionLimit) {
            break;
        }
        link_length = std::nextafter(link_length, 0.0F);
    }

    for (int32_t frame = 0; frame < kFrameCount; ++frame) {
        Pose source = reachable_arm_pose(
            pose_at_frame(fixture.database, frame), link_length);
        write_database_pose(fixture.database, frame, source);
        const Transform object = hand_transform(source);
        write_vec(
            fixture.database.object_positions,
            static_cast<size_t>(frame),
            object.position);
        write_quat(
            fixture.database.object_rotations,
            static_cast<size_t>(frame),
            object.rotation);
    }
    contact_source = pose_at_frame(fixture.database, kContactFrame);
    source_hand = hand_transform(contact_source);
    target_pose = reachable_arm_target(contact_source, angle);
    target_hand = hand_transform(target_pose);

    constexpr float support_height = 0.13F;
    fixture.input.surface = make_surface(
        900U,
        vec3(
            target_hand.position.x,
            target_hand.position.y - support_height,
            target_hand.position.z));
    fixture.input.place_affordance = fixture.input.surface.affordances.front();
    fixture.input.place_affordance.object_in_surface = {
        vec3(0.0F, support_height, 0.0F),
        target_hand.rotation,
    };
    fixture.input.place_affordance.support_point_object = quat_mul_vec3(
        quat_inv(target_hand.rotation),
        vec3(0.0F, -support_height, 0.0F));
    fixture.input.surface.affordances.front() =
        fixture.input.place_affordance;
    fixture.input.ik.accepted_position_m = 0.001F;
    fixture.input.ik.accepted_orientation_radians = 0.005F;
    fixture.input.ik.maximum_iterations = 64;

    const Transform scene = planar_alignment_for_test(
        source_hand, target_hand);
    const Pose start_source = pose_at_frame(
        fixture.database, kHoldFrame + 4);
    fixture.input.current_pose = start_source;
    const Transform mapped_root = compose(
        scene, root_transform(start_source));
    fixture.input.current_pose.positions[kRoot] = mapped_root.position;
    fixture.input.current_pose.rotations[kRoot] = mapped_root.rotation;
    fixture.input.current_object_world = compose(
        scene, hand_transform(start_source));
    return length(
        compose(scene, source_hand).position - target_hand.position);
}

void test_mapped_reverse_hand_correction_uses_exact_ik_limit() {
    PlaceFixture exact = make_fixture(false);
    const float measured = configure_reachable_reverse_request(exact);
    TEST_CHECK(measured <= kPositionLimit);
    TEST_CHECK(kPositionLimit - measured < 0.000001F);
    const PlaceResult exact_result = select_place_motion(exact.input);
    if (!exact_result.accepted) {
        std::cerr << "reachable exact IK rejected reason "
                  << static_cast<int>(exact_result.reason) << '\n';
    }
    TEST_CHECK(exact_result.accepted);
    TEST_CHECK(exact_result.candidate.mode ==
               PlaceMotionMode::ReversedPickup);

    PlaceFixture over = copy_fixture(exact);
    const float lower_surface = std::nextafter(
        over.input.surface.surface_world.position.y,
        -std::numeric_limits<float>::infinity());
    const float delta = lower_surface -
        over.input.surface.surface_world.position.y;
    over.input.surface.surface_world.position.y += delta;
    over.input.surface.support_volume_world.position.y += delta;
    expect_rejected(std::move(over));
}

void advance_until_committed(PlacePlayer& player) {
    for (int tick = 0; tick < 100 && !player.committed(); ++tick) {
        player.advance(0.04F);
    }
    TEST_CHECK(player.committed());
}

void advance_until_release(PlacePlayer& player) {
    for (int tick = 0; tick < 100 && !player.release_due(); ++tick) {
        player.advance(0.04F);
    }
    TEST_CHECK(player.release_due());
}

void test_reverse_player_native_25hz_and_derivative_direction() {
    PlaceFixture fixture = make_fixture(false);
    const PlaceResult selected = select_place_motion(fixture.input);
    TEST_CHECK(selected.accepted);
    PlacePlayer player;
    player.start(selected.candidate, fixture.input);
    TEST_CHECK(player.source_frame() == kStableWindowStop);
    TEST_CHECK(player.phase() == PlacePhase::Align);
    TEST_CHECK(!player.committed());
    TEST_CHECK(!player.finished());

    const Pose source_start = pose_at_frame(
        fixture.database, kStableWindowStop);
    const PlaceSample initial = player.sample();
    TEST_CHECK(initial.source_frame == kStableWindowStop);
    TEST_CHECK(initial.phase == PlacePhase::Align);
    TEST_CHECK(!initial.committed);
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        TEST_CHECK(near(
            initial.pose.velocities[bone],
            -source_start.velocities[bone]));
        TEST_CHECK(near(
            initial.pose.angular_velocities[bone],
            -source_start.angular_velocities[bone]));
    }
    for (size_t dof = 0; dof < initial.pose.hand_dof.size(); ++dof) {
        TEST_CHECK(near(
            initial.pose.hand_dof_velocities[dof],
            -source_start.hand_dof_velocities[dof]));
    }
    TEST_CHECK(initial.pose.foot_contacts == source_start.foot_contacts);

    for (int tick = 1; tick <= 5; ++tick) {
        player.advance(0.04F);
        TEST_CHECK(player.source_frame() == kStableWindowStop - tick);
        TEST_CHECK(player.sample().source_frame == kStableWindowStop - tick);
        TEST_CHECK(player.sample().pose.foot_contacts ==
                   pose_at_frame(
                       fixture.database,
                       kStableWindowStop - tick)
                       .foot_contacts);
    }
    TEST_CHECK(player.source_frame() >= selected.candidate.stop_frame);
}

void test_reverse_player_snaps_fractional_speed_integer_alignment() {
    PlaceFixture fixture = make_fixture(false);
    fixture.input.pickup_candidate.contact_frame = 1;
    fixture.input.pickup_candidate.lift_frame = 12;
    fixture.input.pickup_candidate.hold_frame = 18;
    for (int32_t frame = 1; frame < 12; ++frame) {
        fixture.database.phases[static_cast<size_t>(frame)] =
            static_cast<uint8_t>(Phase::Contact);
    }
    for (int32_t frame = 12; frame < 18; ++frame) {
        fixture.database.phases[static_cast<size_t>(frame)] =
            static_cast<uint8_t>(Phase::Lift);
    }
    for (int32_t frame = 18; frame < kFrameCount; ++frame) {
        fixture.database.phases[static_cast<size_t>(frame)] =
            static_cast<uint8_t>(Phase::Hold);
    }
    for (int32_t frame = 1; frame < kFrameCount - 1; ++frame) {
        fixture.database.hand_contacts[
            static_cast<size_t>(frame) * 2U + 1U] = 1U;
        if (frame > 1) {
            Transform object = database_object_at(fixture.database, frame);
            object.position.y = std::min(
                0.92F,
                0.80F + 0.012F * static_cast<float>(frame - 1));
            set_database_object_and_hand(fixture.database, frame, object);
        }
    }
    fixture.input.timing.playback_speed = 0.85F;
    fixture.input.timing.entry_blend_seconds = 0.04F;
    fixture.input.timing.reversed_commit_seconds = 0.05F;
    const PlaceResult selected = select_place_motion(fixture.input);
    TEST_CHECK(selected.accepted);
    TEST_CHECK(selected.candidate.mode ==
               PlaceMotionMode::ReversedPickup);
    PlacePlayer player;
    player.start(selected.candidate, fixture.input);
    advance_until_committed(player);
    TEST_CHECK(player.source_frame() == selected.candidate.commit_frame);
    for (int tick = 0; tick < 20; ++tick) {
        player.advance(0.04F);
    }
    const int32_t conceptual = selected.candidate.commit_frame - 17;
    TEST_CHECK(conceptual > selected.candidate.release_frame);
    TEST_CHECK(player.source_frame() == conceptual);
    TEST_CHECK(player.sample().source_frame == conceptual);
    TEST_CHECK(near(
        player.sample().pose.positions[kRoot],
        compose(
            selected.candidate.scene_from_source,
            root_transform(pose_at_frame(fixture.database, conceptual)))
            .position));
}

void test_reverse_player_shortest_arc_and_hand_derived_object() {
    PlaceFixture fixture = make_fixture(false);
    write_bone_rotation(
        fixture.database,
        22,
        g1_skeleton::LeftToe,
        quat_from_angle_axis(
            2.967059728F, vec3(0.0F, 1.0F, 0.0F)));
    write_bone_rotation(
        fixture.database,
        23,
        g1_skeleton::LeftToe,
        quat_from_angle_axis(
            -2.967059728F, vec3(0.0F, 1.0F, 0.0F)));
    fixture.database.object_positions[22U * 3U] += 0.01F;
    fixture.input.timing.playback_speed = 0.85F;
    const PlaceResult selected = select_place_motion(fixture.input);
    TEST_CHECK(selected.accepted);
    PlacePlayer player;
    player.start(selected.candidate, fixture.input);
    player.advance(0.04F);
    const Pose left = pose_at_frame(fixture.database, 22);
    const Pose right = pose_at_frame(fixture.database, 23);
    const quat expected = quat_nlerp_shortest(
        left.rotations[g1_skeleton::LeftToe],
        right.rotations[g1_skeleton::LeftToe],
        0.15F);
    const PlaceSample sampled = player.sample();
    TEST_CHECK(near(
        sampled.pose.rotations[g1_skeleton::LeftToe], expected));
    const Transform expected_object = compose(
        hand_transform(sampled.pose),
        inverse(fixture.input.held_affordance.hand_in_object));
    TEST_CHECK(near(sampled.source_object, expected_object));
    const Transform independently_mapped_source = compose(
        selected.candidate.scene_from_source,
        database_object_at(fixture.database, 22));
    TEST_CHECK(!near(
        sampled.source_object,
        independently_mapped_source,
        0.001F));
}

void test_scene_mapping_rotates_only_root_derivatives() {
    PlaceFixture fixture = make_fixture(false);
    const quat yaw = quat_from_angle_axis(
        1.570796327F, vec3(0.0F, 1.0F, 0.0F));
    fixture.input.surface.surface_world.rotation = yaw;
    fixture.input.surface.support_volume_world.rotation = yaw;
    fixture.input.place_affordance.object_in_surface.rotation = quat();
    fixture.input.surface.affordances[0] = fixture.input.place_affordance;
    const PlaceResult selected = select_place_motion(fixture.input);
    TEST_CHECK(selected.accepted);
    if (!selected.accepted) return;
    PlacePlayer player;
    player.start(selected.candidate, fixture.input);
    const Pose source = pose_at_frame(fixture.database, kStableWindowStop);
    const PlaceSample sample = player.sample();
    TEST_CHECK(near(
        sample.pose.velocities[kRoot],
        -quat_mul_vec3(yaw, source.velocities[kRoot])));
    TEST_CHECK(near(
        sample.pose.angular_velocities[kRoot],
        -quat_mul_vec3(yaw, source.angular_velocities[kRoot])));
    TEST_CHECK(near(
        sample.pose.velocities[g1_skeleton::LeftToe],
        -source.velocities[g1_skeleton::LeftToe]));
    TEST_CHECK(near(
        sample.pose.angular_velocities[g1_skeleton::LeftToe],
           -source.angular_velocities[g1_skeleton::LeftToe]));
}

void test_fast_math_half_turn_player_start_rotation_equivalence() {
    PlaceFixture fixture = make_fixture(false);
    const quat half_turn(0.0F, 0.0F, 1.0F, 0.0F);
    fixture.input.surface.surface_world.rotation = half_turn;
    fixture.input.surface.support_volume_world.rotation = half_turn;
    fixture.input.place_affordance.object_in_surface.rotation = quat();
    fixture.input.surface.affordances[0] = fixture.input.place_affordance;

    const PlaceResult selected = select_place_motion(fixture.input);
    TEST_CHECK(selected.accepted);
    if (!selected.accepted) return;

    PlacePlayer identical;
    TEST_CHECK(!throws_as<std::invalid_argument>([&] {
        identical.start(selected.candidate, fixture.input);
    }));

    PlaceCandidate antipodal = selected.candidate;
    antipodal.scene_from_source.rotation =
        -antipodal.scene_from_source.rotation;
    antipodal.staging_root_world.rotation =
        -antipodal.staging_root_world.rotation;
    PlacePlayer equivalent;
    TEST_CHECK(!throws_as<std::invalid_argument>([&] {
        equivalent.start(antipodal, fixture.input);
    }));
}

void assert_release_clamp(float speed, bool recorded) {
    PlaceFixture fixture = make_fixture(recorded);
    if (!recorded) remove_recorded(fixture);
    fixture.input.timing.playback_speed = speed;
    const PlaceResult selected = select_place_motion(fixture.input);
    TEST_CHECK(selected.accepted);
    TEST_CHECK(selected.candidate.mode == (recorded
        ? PlaceMotionMode::RecordedPlace
        : PlaceMotionMode::ReversedPickup));
    PlacePlayer player;
    player.start(selected.candidate, fixture.input);
    if (!recorded) {
        const Pose source = pose_at_frame(
            fixture.database, selected.candidate.entry_frame);
        const PlaceSample initial = player.sample();
        TEST_CHECK(near(
            initial.pose.velocities[g1_skeleton::LeftToe],
            -speed * source.velocities[g1_skeleton::LeftToe]));
        TEST_CHECK(near(
            initial.pose.angular_velocities[g1_skeleton::LeftToe],
            -speed * source.angular_velocities[g1_skeleton::LeftToe]));
        TEST_CHECK(near(
            initial.pose.hand_dof_velocities[3],
            -speed * source.hand_dof_velocities[3]));
    }
    advance_until_committed(player);
    TEST_CHECK(player.source_frame() == selected.candidate.commit_frame);
    TEST_CHECK(player.phase() == PlacePhase::Lower);
    TEST_CHECK(player.sample().committed);
    advance_until_release(player);
    TEST_CHECK(player.source_frame() == selected.candidate.release_frame);
    TEST_CHECK(player.phase() == PlacePhase::Release);
    const PlaceSample release = player.sample();
    TEST_CHECK(release.source_frame == selected.candidate.release_frame);
    TEST_CHECK(release.phase == PlacePhase::Release);
    TEST_CHECK(release.committed);

    const PlaceSample repeated = player.sample();
    TEST_CHECK(repeated.source_frame == release.source_frame);
    TEST_CHECK(near(repeated.pose.positions[kRoot], release.pose.positions[kRoot]));
    player.advance(0.04F);
    TEST_CHECK(player.release_due());
    TEST_CHECK(player.source_frame() == selected.candidate.release_frame);
    TEST_CHECK(near(
        player.sample().pose.positions[kRoot], release.pose.positions[kRoot]));

    player.acknowledge_release();
    TEST_CHECK(!player.release_due());
    TEST_CHECK(player.phase() == PlacePhase::Retract);
    player.advance(0.04F);
    TEST_CHECK(!player.release_due());
    TEST_CHECK(player.phase() == PlacePhase::Retract || player.finished());
    TEST_CHECK(!near(
        player.sample().pose.positions[kRoot],
        release.pose.positions[kRoot],
        1.0e-7F) || player.finished());

    for (int tick = 0; tick < 100 && !player.finished(); ++tick) {
        player.advance(0.04F);
        TEST_CHECK(!player.release_due());
    }
    TEST_CHECK(player.finished());
    TEST_CHECK(player.source_frame() == selected.candidate.stop_frame);
    TEST_CHECK(player.phase() == PlacePhase::Finished);
}

void test_all_speeds_clamp_commit_and_release_once() {
    for (float speed : {0.85F, 1.0F, 1.15F}) {
        assert_release_clamp(speed, false);
        assert_release_clamp(speed, true);
    }
}

void test_recorded_player_maps_authored_object_and_positive_derivatives() {
    PlaceFixture fixture = make_fixture();
    fixture.input.timing.playback_speed = 1.15F;
    const PlaceResult selected = select_place_motion(fixture.input);
    TEST_CHECK(selected.accepted);
    PlacePlayer player;
    player.start(selected.candidate, fixture.input);
    const RecordedPlaceClip& clip = fixture.library.recorded.front();
    const Pose& source = clip.poses[static_cast<size_t>(clip.entry_frame)];
    const PlaceSample sample = player.sample();
    TEST_CHECK(sample.source_frame == clip.entry_frame);
    TEST_CHECK(near(
        sample.source_object,
        compose(
            selected.candidate.scene_from_source,
            clip.object_poses[static_cast<size_t>(clip.entry_frame)])));
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        const vec3 expected_velocity = bone == kRoot
            ? 1.15F * quat_mul_vec3(
                selected.candidate.scene_from_source.rotation,
                source.velocities[bone])
            : 1.15F * source.velocities[bone];
        const vec3 expected_angular = bone == kRoot
            ? 1.15F * quat_mul_vec3(
                selected.candidate.scene_from_source.rotation,
                source.angular_velocities[bone])
            : 1.15F * source.angular_velocities[bone];
        TEST_CHECK(near(sample.pose.velocities[bone], expected_velocity));
        TEST_CHECK(near(
            sample.pose.angular_velocities[bone], expected_angular));
    }
}

void test_player_start_and_fixed_tick_validation_are_atomic() {
    PlaceFixture fixture = make_fixture(false);
    const PlaceResult selected = select_place_motion(fixture.input);
    TEST_CHECK(selected.accepted);
    PlacePlayer player;
    PlaceCandidate forged = selected.candidate;
    forged.selection_id ^= 1U;
    TEST_CHECK(throws_as<std::invalid_argument>([&] {
        player.start(forged, fixture.input);
    }));
    TEST_CHECK(player.finished());

    PlaceFixture stale = copy_fixture(fixture);
    stale.input.current_pose.hand_dof[0] += 0.001F;
    TEST_CHECK(throws_as<std::invalid_argument>([&] {
        player.start(selected.candidate, stale.input);
    }));
    TEST_CHECK(player.finished());

    player.start(selected.candidate, fixture.input);
    const PlaceSample before = player.sample();
    for (float invalid : {
             0.0F,
             -0.04F,
             0.02F,
             std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::quiet_NaN()}) {
        TEST_CHECK(throws_as<std::invalid_argument>([&] {
            player.advance(invalid);
        }));
        TEST_CHECK(player.source_frame() == before.source_frame);
        TEST_CHECK(near(
            player.sample().pose.positions[kRoot],
            before.pose.positions[kRoot]));
    }
    TEST_CHECK(throws_as<std::logic_error>([&] {
        player.acknowledge_release();
    }));
    TEST_CHECK(player.source_frame() == before.source_frame);
}

void test_fast_math_bit_safe_canary() {
    using IKFloatMember = float IKConfig::*;
    constexpr std::array<IKFloatMember, 8> fields = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (IKFloatMember field : fields) {
        for (float invalid : {
                 std::numeric_limits<float>::quiet_NaN(),
                 std::numeric_limits<float>::infinity()}) {
            PlaceFixture fixture = make_fixture(false);
            fixture.input.ik.*field = invalid;
            const PlaceResult result = select_place_motion(fixture.input);
            TEST_CHECK(!result.accepted);
            TEST_CHECK(result.reason == Reason::CorrectionLimit);
        }
    }
    PlaceFixture negative_iterations = make_fixture(false);
    negative_iterations.input.ik.maximum_iterations = -1;
    TEST_CHECK(!select_place_motion(negative_iterations.input).accepted);

    PlaceFixture invalid_pose = make_fixture(false);
    invalid_pose.input.current_pose.positions[kRoot].x =
        std::numeric_limits<float>::quiet_NaN();
    const PlaceResult pose_result = select_place_motion(invalid_pose.input);
    TEST_CHECK(!pose_result.accepted);
    TEST_CHECK(pose_result.reason == Reason::TargetUnavailable);

    constexpr float orientation_limit = 0.10F;
    PlaceFixture exact_recorded = make_fixture();
    set_clip_grasp(
        exact_recorded.library.recorded.front(),
        Transform{
            vec3(),
            quat_from_angle_axis(
                orientation_limit, vec3(0.0F, 1.0F, 0.0F))});
    exact_recorded.input.ik.maximum_request_orientation_radians =
        orientation_limit;
    exact_recorded.input.ik.accepted_orientation_radians = orientation_limit;
    exact_recorded.input.ik.maximum_iterations = 0;
    const PlaceResult exact_tier = select_place_motion(exact_recorded.input);
    TEST_CHECK(exact_tier.accepted);
    TEST_CHECK(exact_tier.candidate.mode == PlaceMotionMode::RecordedPlace);

    PlaceFixture above_recorded = make_fixture();
    set_clip_grasp(
        above_recorded.library.recorded.front(),
        Transform{
            vec3(),
            quat_from_angle_axis(
                std::nextafter(
                    orientation_limit,
                    std::numeric_limits<float>::infinity()),
                vec3(0.0F, 1.0F, 0.0F))});
    above_recorded.input.ik.maximum_request_orientation_radians =
        orientation_limit;
    above_recorded.input.ik.accepted_orientation_radians = orientation_limit;
    above_recorded.input.ik.maximum_iterations = 0;
    const PlaceResult above_tier = select_place_motion(above_recorded.input);
    TEST_CHECK(above_tier.accepted);
    TEST_CHECK(above_tier.candidate.mode == PlaceMotionMode::ReversedPickup);

    PlaceFixture boundary = make_fixture(false);
    boundary.input.timing.entry_blend_seconds = 0.20F;
    boundary.input.timing.reversed_commit_seconds = 0.20F;
    boundary.input.timing.maximum_alignment_seconds = 0.30F;
    const PlaceResult selected = select_place_motion(boundary.input);
    TEST_CHECK(selected.accepted);
    if (!selected.accepted) return;
    PlacePlayer player;
    player.start(selected.candidate, boundary.input);
    const int32_t before = player.source_frame();
    for (float invalid : {
             std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity(),
             std::nextafter(0.04F, 0.0F),
             std::nextafter(
                 0.04F, std::numeric_limits<float>::infinity())}) {
        TEST_CHECK(throws_as<std::invalid_argument>([&] {
            player.advance(invalid);
        }));
        TEST_CHECK(player.source_frame() == before);
    }
    test_fast_math_half_turn_player_start_rotation_equivalence();
    test_scene_mapping_rotates_only_root_derivatives();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 &&
        std::string_view(argv[1]) == "--fast-math-canary") {
        test_fast_math_bit_safe_canary();
        return g_failures == 0 ? 0 : 1;
    }
    test_frozen_public_contract();
    test_recorded_priority_and_dynamic_readiness();
    test_readiness_boundaries_and_tighter_config();
    test_invalid_timing_and_match_configs_reject_before_tiers();
    test_ik_config_validation_and_authoritative_request_limits();
    test_reverse_certification_uses_earliest_stable_prefix();
    test_reverse_identity_ignores_unconsulted_storage_and_clips();
    test_reverse_certification_rejections();
    test_recorded_rows_validate_locally_before_unique_priority();
    test_recorded_ranking_uses_aligned_entry_pose_continuity();
    test_recorded_structural_physics_and_compatibility_rejections();
    test_recorded_release_support_boundaries();
    test_selection_certifies_mapped_support_volume_clearance();
    test_selection_runs_real_ik_with_exact_config();
    test_placement_fit_failure_reasons_are_preserved();
    test_complete_snapshot_identity_and_pointer_independence();
    test_ik_fingerprint_is_complete_and_configuration_only();
    test_canonical_identity_normalizes_negative_zero_and_quaternion_sign();
    test_commit_event_derivation_and_runtime_bounds();
    test_mapped_reverse_hand_correction_uses_exact_ik_limit();
    test_reverse_player_native_25hz_and_derivative_direction();
    test_reverse_player_snaps_fractional_speed_integer_alignment();
    test_reverse_player_shortest_arc_and_hand_derived_object();
    test_scene_mapping_rotates_only_root_derivatives();
    test_all_speeds_clamp_commit_and_release_once();
    test_recorded_player_maps_authored_object_and_positive_derivatives();
    test_player_start_and_fixed_tick_validation_are_atomic();
    return g_failures == 0 ? 0 : 1;
}
