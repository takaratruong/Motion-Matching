#include "interaction_place_controller.h"
#include "interaction_rotation_gate.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace interaction;

constexpr int32_t kFrameCount = 30;
constexpr int32_t kContactFrame = 8;
constexpr int32_t kLiftFrame = 12;
constexpr int32_t kHoldFrame = 18;
constexpr uint64_t kProfileId = 3001U;
constexpr size_t kRoot = static_cast<size_t>(g1_skeleton::Simulation);
constexpr size_t kRightHand = static_cast<size_t>(g1_skeleton::RightWrist);
constexpr float kPositionCap = 0.12F;
constexpr float kOrientationCap = 0.436332313F;
constexpr float kReleaseOrientationCap = 0.174532925F;

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
    if (quat_dot(left, right) < 0.0F) right = -right;
    return near(left.w, right.w, tolerance) &&
           near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near(Transform left, Transform right, float tolerance = 1.0e-5F) {
    return near(left.position, right.position, tolerance) &&
           near(left.rotation, right.rotation, tolerance);
}

bool exactly_equal(Transform left, Transform right) {
    return left.position.x == right.position.x &&
           left.position.y == right.position.y &&
           left.position.z == right.position.z &&
           left.rotation.w == right.rotation.w &&
           left.rotation.x == right.rotation.x &&
           left.rotation.y == right.rotation.y &&
           left.rotation.z == right.rotation.z;
}

bool near(const Pose& left, const Pose& right, float tolerance = 1.0e-5F) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (!near(left.positions[bone], right.positions[bone], tolerance) ||
            !near(left.velocities[bone], right.velocities[bone], tolerance) ||
            !near(left.rotations[bone], right.rotations[bone], tolerance) ||
            !near(
                left.angular_velocities[bone],
                right.angular_velocities[bone],
                tolerance)) {
            return false;
        }
    }
    for (size_t dof = 0; dof < left.hand_dof.size(); ++dof) {
        if (!near(left.hand_dof[dof], right.hand_dof[dof], tolerance) ||
            !near(
                left.hand_dof_velocities[dof],
                right.hand_dof_velocities[dof],
                tolerance)) {
            return false;
        }
    }
    return left.foot_contacts == right.foot_contacts;
}

bool equivalent(const PlaceStep& left, const PlaceStep& right) {
    return near(left.pose, right.pose) &&
           near(left.object_world, right.object_world) &&
           left.phase == right.phase &&
           left.source_frame == right.source_frame &&
           left.source_frame_exact == right.source_frame_exact &&
           left.committed == right.committed &&
           left.release_due == right.release_due &&
           left.retract_finished == right.retract_finished &&
           left.recover_to_carry == right.recover_to_carry &&
           left.reason == right.reason &&
           near(
               left.hand_position_error_m,
               right.hand_position_error_m) &&
           near(
               left.hand_orientation_error_radians,
               right.hand_orientation_error_radians) &&
           near(
               left.requested_root_correction_m,
               right.requested_root_correction_m) &&
           near(
               left.applied_root_correction_m,
               right.applied_root_correction_m) &&
           near(
               left.requested_yaw_correction_radians,
               right.requested_yaw_correction_radians) &&
           near(
               left.applied_yaw_correction_radians,
               right.applied_yaw_correction_radians) &&
           near(
               left.requested_hand_correction_m,
               right.requested_hand_correction_m) &&
           near(
               left.applied_hand_correction_m,
               right.applied_hand_correction_m) &&
           near(
               left.requested_hand_orientation_radians,
               right.requested_hand_orientation_radians) &&
           near(
               left.applied_hand_orientation_radians,
               right.applied_hand_orientation_radians) &&
           left.actual_fit.accepted == right.actual_fit.accepted &&
           left.actual_fit.reason == right.actual_fit.reason &&
           near(
               left.actual_fit.support_gap_m,
               right.actual_fit.support_gap_m) &&
           near(
               left.actual_fit.lowest_corner_m,
               right.actual_fit.lowest_corner_m) &&
           near(
               left.actual_fit.highest_corner_m,
               right.actual_fit.highest_corner_m) &&
           left.actual_fit.footprint_valid ==
               right.actual_fit.footprint_valid &&
           left.actual_fit.overhead_valid == right.actual_fit.overhead_valid &&
           left.support_sweep_clear == right.support_sweep_clear;
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

void write_pose(Database& database, int32_t frame, const Pose& pose) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        const size_t vector = vec_offset(frame, bone);
        database.positions.at(vector) = pose.positions[bone].x;
        database.positions.at(vector + 1U) = pose.positions[bone].y;
        database.positions.at(vector + 2U) = pose.positions[bone].z;
        database.velocities.at(vector) = pose.velocities[bone].x;
        database.velocities.at(vector + 1U) = pose.velocities[bone].y;
        database.velocities.at(vector + 2U) = pose.velocities[bone].z;
        database.angular_velocities.at(vector) =
            pose.angular_velocities[bone].x;
        database.angular_velocities.at(vector + 1U) =
            pose.angular_velocities[bone].y;
        database.angular_velocities.at(vector + 2U) =
            pose.angular_velocities[bone].z;
        const size_t rotation = quat_offset(frame, bone);
        database.rotations.at(rotation) = pose.rotations[bone].w;
        database.rotations.at(rotation + 1U) = pose.rotations[bone].x;
        database.rotations.at(rotation + 2U) = pose.rotations[bone].y;
        database.rotations.at(rotation + 3U) = pose.rotations[bone].z;
    }
    for (size_t dof = 0; dof < pose.hand_dof.size(); ++dof) {
        const size_t offset = static_cast<size_t>(frame) * 14U + dof;
        database.hand_dof.at(offset) = pose.hand_dof[dof];
        database.hand_dof_velocities.at(offset) =
            pose.hand_dof_velocities[dof];
    }
    database.foot_contacts.at(static_cast<size_t>(frame) * 2U) =
        pose.foot_contacts[0];
    database.foot_contacts.at(static_cast<size_t>(frame) * 2U + 1U) =
        pose.foot_contacts[1];
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

Transform hand_derived_object(
    const Pose& pose,
    const GraspAffordance& affordance) {
    return compose(
        hand_transform(pose, affordance.hand),
        inverse(affordance.hand_in_object));
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
        pose.velocities[bone] = vec3(0.10F + marker, 0.20F, 0.30F);
        pose.angular_velocities[bone] = vec3(0.40F, 0.50F + marker, 0.60F);
    }
    pose.positions[kRoot] = root;
    pose.positions[kRightHand] = hand.position - root;
    pose.rotations[kRightHand] = hand.rotation;
    for (size_t dof = 0; dof < pose.hand_dof.size(); ++dof) {
        pose.hand_dof[dof] = marker + 0.01F * static_cast<float>(dof);
        pose.hand_dof_velocities[dof] = 0.7F + marker;
    }
    pose.foot_contacts = {0U, 1U};
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
        if (frame >= kContactFrame && frame < kLiftFrame) phase = Phase::Contact;
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
                  0.80F + 0.012F * static_cast<float>(frame - kContactFrame));
        const Transform object{vec3(0.0F, object_height, 0.0F), quat()};
        Transform hand = object;
        if (frame == kHoldFrame) hand.position.x += 0.020001F;
        const Pose pose = make_pose(
            root, hand, 0.001F * static_cast<float>(frame));
        write_pose(database, frame, pose);
        if (frame >= kContactFrame && frame < kFrameCount - 1) {
            database.hand_contacts[static_cast<size_t>(frame) * 2U + 1U] = 1U;
        }
        write_vec(database.object_positions, static_cast<size_t>(frame), object.position);
        write_quat(database.object_rotations, static_cast<size_t>(frame), object.rotation);
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

struct Fixture {
    Database database{};
    PlaceMotionLibrary library{};
    PlaceMatchInput input{};
};

void refresh(Fixture& fixture) {
    fixture.input.pickup_database = &fixture.database;
    fixture.input.library = &fixture.library;
}

Fixture make_fixture(bool recorded = true) {
    Fixture fixture{};
    fixture.database = make_reverse_database();
    if (recorded) fixture.library.recorded.push_back(make_recorded_clip());
    refresh(fixture);
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
        {},
    };
    fixture.input.surface = make_surface();
    fixture.input.place_affordance = fixture.input.surface.affordances.front();
    fixture.input.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    const RecordedPlaceClip reference = make_recorded_clip();
    const Transform goal = placement_goal_world(
        fixture.input.surface,
        fixture.input.place_affordance.object_in_surface);
    const Transform scene = compose(
        goal,
        inverse(reference.object_poses[static_cast<size_t>(
            reference.release_frame)]));
    fixture.input.current_pose = reference.poses.front();
    const Transform mapped_root = compose(
        scene, root_transform(reference.poses.front()));
    fixture.input.current_pose.positions[kRoot] = mapped_root.position;
    fixture.input.current_pose.rotations[kRoot] = mapped_root.rotation;
    fixture.input.current_object_world = compose(scene, reference.object_poses.front());
    return fixture;
}

Fixture copy_fixture(const Fixture& source) {
    Fixture copy = source;
    refresh(copy);
    return copy;
}

void restage_fixture(Fixture& fixture) {
    const RecordedPlaceClip reference = make_recorded_clip();
    const Transform goal = placement_goal_world(
        fixture.input.surface,
        fixture.input.place_affordance.object_in_surface);
    const Transform scene = compose(
        goal,
        inverse(reference.object_poses[static_cast<size_t>(
            reference.release_frame)]));
    fixture.input.current_pose = reference.poses.front();
    const Transform mapped_root = compose(
        scene, root_transform(reference.poses.front()));
    fixture.input.current_pose.positions[kRoot] = mapped_root.position;
    fixture.input.current_pose.rotations[kRoot] = mapped_root.rotation;
    fixture.input.current_object_world = compose(scene, reference.object_poses.front());
    refresh(fixture);
}

void stage_from_recorded_entry(Fixture& fixture) {
    const RecordedPlaceClip& clip = fixture.library.recorded.front();
    const Transform goal = placement_goal_world(
        fixture.input.surface,
        fixture.input.place_affordance.object_in_surface);
    const Transform scene = compose(
        goal,
        inverse(clip.object_poses[static_cast<size_t>(clip.release_frame)]));
    fixture.input.current_pose = clip.poses[static_cast<size_t>(clip.entry_frame)];
    const Transform mapped_root = compose(
        scene, root_transform(fixture.input.current_pose));
    fixture.input.current_pose.positions[kRoot] = mapped_root.position;
    fixture.input.current_pose.rotations[kRoot] = mapped_root.rotation;
    fixture.input.current_object_world = compose(
        scene, clip.object_poses[static_cast<size_t>(clip.entry_frame)]);
    refresh(fixture);
}

void map_current_rigidly(Fixture& fixture, Transform target_root) {
    const Transform current_root = root_transform(fixture.input.current_pose);
    const Transform mapping = compose(target_root, inverse(current_root));
    fixture.input.current_pose.positions[kRoot] = target_root.position;
    fixture.input.current_pose.rotations[kRoot] = target_root.rotation;
    fixture.input.current_object_world = compose(
        mapping, fixture.input.current_object_world);
}

void set_recorded_grasp(RecordedPlaceClip& clip, Transform hand_in_object) {
    clip.hand_in_object = hand_in_object;
    for (size_t frame = 0; frame < clip.poses.size(); ++frame) {
        Pose& pose = clip.poses[frame];
        const Transform hand = compose(clip.object_poses[frame], hand_in_object);
        pose.positions[kRightHand] = hand.position - pose.positions[kRoot];
        pose.rotations[kRightHand] = hand.rotation;
    }
}

Transform mapped_recorded_hand(
    const RecordedPlaceClip& clip,
    const PlaceCandidate& candidate) {
    Pose pose = clip.poses[static_cast<size_t>(clip.release_frame)];
    const Transform mapped_root = compose(
        candidate.scene_from_source,
        Transform{pose.positions[kRoot], pose.rotations[kRoot]});
    pose.positions[kRoot] = mapped_root.position;
    pose.rotations[kRoot] = mapped_root.rotation;
    return hand_transform(pose, clip.hand);
}

double transform_distance(vec3 left, vec3 right) {
    const double x = static_cast<double>(left.x) - right.x;
    const double y = static_cast<double>(left.y) - right.y;
    const double z = static_cast<double>(left.z) - right.z;
    return std::sqrt(x * x + y * y + z * z);
}

double transform_rotation_error(quat left, quat right) {
    const rotation_gate::Measure measured = rotation_gate::measure(
        left, right);
    return measured.valid
        ? measured.radians
        : std::numeric_limits<double>::max();
}

void move_release_hand_to_first_position_above(
    RecordedPlaceClip& clip,
    const PlaceCandidate& candidate,
    Transform goal_hand,
    float limit) {
    Pose& release = clip.poses[static_cast<size_t>(clip.release_frame)];
    float value = release.positions[kRightHand].x;
    for (int step = 0; step < 4096; ++step) {
        value = std::nextafter(value, std::numeric_limits<float>::infinity());
        release.positions[kRightHand].x = value;
        if (transform_distance(
                mapped_recorded_hand(clip, candidate).position,
                goal_hand.position) > limit) {
            return;
        }
    }
    throw std::logic_error("could not encode position nextabove boundary");
}

void move_release_hand_to_first_orientation_above(
    RecordedPlaceClip& clip,
    const PlaceCandidate& candidate,
    Transform goal_hand,
    float limit) {
    Pose& release = clip.poses[static_cast<size_t>(clip.release_frame)];
    float angle = limit;
    for (int step = 0; step < 4096; ++step) {
        angle = std::nextafter(
            angle, std::numeric_limits<float>::infinity());
        release.rotations[kRightHand] = quat_from_angle_axis(
            angle, vec3(0.0F, 1.0F, 0.0F));
        if (transform_rotation_error(
                mapped_recorded_hand(clip, candidate).rotation,
                goal_hand.rotation) > limit) {
            return;
        }
    }
    throw std::logic_error("could not encode orientation nextabove boundary");
}

void set_release_position_error_boundary(
    RecordedPlaceClip& clip,
    const PlaceCandidate& candidate,
    Transform goal_hand,
    float limit,
    bool above) {
    Pose& release = clip.poses[static_cast<size_t>(clip.release_frame)];
    const float original = release.positions[kRightHand].x;
    float below_value = original + limit;
    float above_value = below_value;
    bool have_below = false;
    bool have_above = false;
    float best_below = below_value;
    float best_above = above_value;
    double best_below_error = -1.0;
    double best_above_error = std::numeric_limits<double>::infinity();
    for (int step = 0; step < 4096; ++step) {
        for (float value : {below_value, above_value}) {
            release.positions[kRightHand].x = value;
            const double error = transform_distance(
                mapped_recorded_hand(clip, candidate).position,
                goal_hand.position);
            if (error <= limit && error > best_below_error) {
                have_below = true;
                best_below = value;
                best_below_error = error;
            }
            if (error > limit && error < best_above_error) {
                have_above = true;
                best_above = value;
                best_above_error = error;
            }
        }
        below_value = std::nextafter(
            below_value, -std::numeric_limits<float>::infinity());
        above_value = std::nextafter(
            above_value, std::numeric_limits<float>::infinity());
    }
    if ((above && !have_above) || (!above && !have_below)) {
        throw std::logic_error("could not encode release position boundary");
    }
    release.positions[kRightHand].x = above ? best_above : best_below;
}

void set_release_orientation_error_boundary(
    RecordedPlaceClip& clip,
    const PlaceCandidate& candidate,
    Transform goal_hand,
    float limit,
    bool above) {
    Pose& release = clip.poses[static_cast<size_t>(clip.release_frame)];
    float below_angle = limit;
    float above_angle = limit;
    bool have_below = false;
    bool have_above = false;
    quat best_below{};
    quat best_above{};
    double best_below_error = -1.0;
    double best_above_error = std::numeric_limits<double>::infinity();
    for (int step = 0; step < 4096; ++step) {
        for (float angle : {below_angle, above_angle}) {
            const quat rotation = quat_from_angle_axis(
                angle, vec3(0.0F, 1.0F, 0.0F));
            release.rotations[kRightHand] = rotation;
            const double error = transform_rotation_error(
                mapped_recorded_hand(clip, candidate).rotation,
                goal_hand.rotation);
            if (error <= limit && error > best_below_error) {
                have_below = true;
                best_below = rotation;
                best_below_error = error;
            }
            if (error > limit && error < best_above_error) {
                have_above = true;
                best_above = rotation;
                best_above_error = error;
            }
        }
        below_angle = std::nextafter(
            below_angle, -std::numeric_limits<float>::infinity());
        above_angle = std::nextafter(
            above_angle, std::numeric_limits<float>::infinity());
    }
    if ((above && !have_above) || (!above && !have_below)) {
        throw std::logic_error("could not encode release orientation boundary");
    }
    release.rotations[kRightHand] = above ? best_above : best_below;
}

PlaceBeginInput selected_begin(Fixture& fixture) {
    refresh(fixture);
    const PlaceResult selected = select_place_motion(fixture.input);
    TEST_CHECK(selected.accepted);
    return {fixture.input, selected.candidate};
}

PlaceControllerConfig controller_config(const Fixture& fixture) {
    PlaceControllerConfig config{};
    config.timing = fixture.input.timing;
    config.match = fixture.input.match;
    return config;
}

PlaceController make_controller(const Fixture& fixture) {
    return PlaceController(controller_config(fixture), fixture.input.ik);
}

void shift_recorded_hand(RecordedPlaceClip& clip, int32_t frame, vec3 delta);
PlaceStep run_to_terminal_before_ack(PlaceController& controller);

static_assert(!std::is_default_constructible_v<PlaceController>);
static_assert(std::is_constructible_v<
              PlaceController,
              PlaceControllerConfig,
              IKConfig>);

void test_frozen_public_contract() {
    const PlaceControllerConfig config{};
    TEST_CHECK(config.timing.canonical_fps == 25.0F);
    TEST_CHECK(config.timing.playback_speed == 1.0F);
    TEST_CHECK(config.timing.entry_blend_seconds == 0.25F);
    TEST_CHECK(config.timing.reversed_commit_seconds == 0.50F);
    TEST_CHECK(config.timing.maximum_alignment_seconds == 1.00F);
    TEST_CHECK(config.match.maximum_entry_root_error_m == 0.25F);
    TEST_CHECK(config.match.maximum_entry_yaw_error_radians == 0.436332313F);
    TEST_CHECK(config.release_position_m == 0.02F);
    TEST_CHECK(config.release_orientation_radians == kReleaseOrientationCap);

    PlaceBeginInput begin{};
    PlaceBeginResult begin_result{};
    PlaceStep step{};
    TEST_CHECK(!begin_result.accepted);
    TEST_CHECK(begin_result.reason == Reason::None);
    TEST_CHECK(step.phase == PlacePhase::Align);
    TEST_CHECK(step.source_frame_exact == -1.0);
    TEST_CHECK(step.requested_root_correction_m == 0.0F);
    TEST_CHECK(step.applied_root_correction_m == 0.0F);
    TEST_CHECK(step.requested_yaw_correction_radians == 0.0F);
    TEST_CHECK(step.applied_yaw_correction_radians == 0.0F);
    TEST_CHECK(step.requested_hand_correction_m == 0.0F);
    TEST_CHECK(step.applied_hand_correction_m == 0.0F);
    TEST_CHECK(step.requested_hand_orientation_radians == 0.0F);
    TEST_CHECK(step.applied_hand_orientation_radians == 0.0F);
    (void)begin;
}

void test_rotation_gate_is_scale_sign_and_boundary_stable() {
    constexpr float limit = 0.10F;
    const quat exact = quat_from_angle_axis(
        limit, vec3(0.0F, 1.0F, 0.0F));
    const quat above = quat_from_angle_axis(
        std::nextafter(limit, std::numeric_limits<float>::infinity()),
        vec3(0.0F, 1.0F, 0.0F));
    TEST_CHECK(rotation_gate::within(exact, quat(), limit));
    TEST_CHECK(!rotation_gate::within(above, quat(), limit));
    TEST_CHECK(rotation_gate::within(-exact, quat(), limit));
    TEST_CHECK(rotation_gate::within(4.0F * exact, 2.0F * quat(), limit));
    TEST_CHECK(!rotation_gate::within(4.0F * above, 2.0F * quat(), limit));

    constexpr float pi = 3.141592654F;
    const float near_pi = std::nextafter(pi, 0.0F);
    const quat near_pi_exact = quat_from_angle_axis(
        near_pi, vec3(1.0F, 0.0F, 0.0F));
    const quat half_turn = quat_from_angle_axis(
        pi, vec3(1.0F, 0.0F, 0.0F));
    TEST_CHECK(rotation_gate::within(near_pi_exact, quat(), near_pi));
    TEST_CHECK(!rotation_gate::within(half_turn, quat(), near_pi));

    quat invalid = exact;
    invalid.w = std::numeric_limits<float>::quiet_NaN();
    TEST_CHECK(!rotation_gate::within(invalid, quat(), limit));
    const rotation_gate::Measure positive = rotation_gate::measure(
        exact, quat());
    const rotation_gate::Measure antipodal = rotation_gate::measure(
        -exact, quat());
    TEST_CHECK(positive.valid);
    TEST_CHECK(antipodal.valid);
    TEST_CHECK(positive.radians == antipodal.radians);
}

void test_constructor_validates_every_configuration_family() {
    const Fixture fixture = make_fixture();
    const PlaceControllerConfig valid = controller_config(fixture);
    const IKConfig valid_ik = fixture.input.ik;
    TEST_CHECK(!throws_as<std::invalid_argument>([&] {
        PlaceController controller(valid, valid_ik);
        (void)controller;
    }));

    const auto rejects_config = [&](PlaceControllerConfig config) {
        TEST_CHECK(throws_as<std::invalid_argument>([&] {
            PlaceController controller(config, valid_ik);
            (void)controller;
        }));
    };
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();

    for (float value : {0.0F, nan, infinity,
                        std::nextafter(25.0F, infinity)}) {
        PlaceControllerConfig config = valid;
        config.timing.canonical_fps = value;
        rejects_config(config);
    }
    for (float value : {0.0F, nan, infinity,
                        std::nextafter(0.85F, -infinity),
                        std::nextafter(1.15F, infinity)}) {
        PlaceControllerConfig config = valid;
        config.timing.playback_speed = value;
        rejects_config(config);
    }
    for (float value : {0.0F, nan, infinity, 1.01F}) {
        PlaceControllerConfig config = valid;
        config.timing.entry_blend_seconds = value;
        rejects_config(config);
        config = valid;
        config.timing.reversed_commit_seconds = value;
        rejects_config(config);
        config = valid;
        config.timing.maximum_alignment_seconds = value;
        rejects_config(config);
    }
    {
        PlaceControllerConfig config = valid;
        config.timing.maximum_alignment_seconds =
            std::nextafter(config.timing.entry_blend_seconds, 0.0F);
        rejects_config(config);
        config = valid;
        config.timing.maximum_alignment_seconds =
            std::nextafter(config.timing.reversed_commit_seconds, 0.0F);
        rejects_config(config);
    }
    for (float value : {0.0F, nan, infinity,
                        std::nextafter(0.25F, infinity)}) {
        PlaceControllerConfig config = valid;
        config.match.maximum_entry_root_error_m = value;
        rejects_config(config);
    }
    for (float value : {0.0F, nan, infinity,
                        std::nextafter(kOrientationCap, infinity)}) {
        PlaceControllerConfig config = valid;
        config.match.maximum_entry_yaw_error_radians = value;
        rejects_config(config);
    }
    for (float value : {0.0F, -0.01F, nan, infinity,
                        std::nextafter(0.02F, infinity)}) {
        PlaceControllerConfig config = valid;
        config.release_position_m = value;
        rejects_config(config);
    }
    for (float value : {0.0F, -0.01F, nan, infinity,
                        std::nextafter(kReleaseOrientationCap, infinity)}) {
        PlaceControllerConfig config = valid;
        config.release_orientation_radians = value;
        rejects_config(config);
    }

    const auto rejects_ik = [&](IKConfig ik) {
        TEST_CHECK(throws_as<std::invalid_argument>([&] {
            PlaceController controller(valid, ik);
            (void)controller;
        }));
    };
    using Member = float IKConfig::*;
    const std::array<Member, 8> fields = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (size_t index = 0; index < fields.size(); ++index) {
        for (float value : {nan, infinity, -0.01F}) {
            IKConfig ik = valid_ik;
            ik.*fields[index] = value;
            rejects_ik(ik);
        }
        if (index >= 4U) {
            IKConfig ik = valid_ik;
            ik.*fields[index] = 0.0F;
            rejects_ik(ik);
        }
    }
    {
        IKConfig ik = valid_ik;
        ik.maximum_iterations = -1;
        rejects_ik(ik);
        ik = valid_ik;
        ik.maximum_request_position_m = std::nextafter(kPositionCap, infinity);
        rejects_ik(ik);
        ik = valid_ik;
        ik.maximum_request_orientation_radians =
            std::nextafter(kOrientationCap, infinity);
        rejects_ik(ik);
    }
    for (float tighter : {0.0F, 0.01F}) {
        IKConfig ik = valid_ik;
        ik.maximum_request_position_m = tighter;
        ik.maximum_request_orientation_radians = tighter;
        TEST_CHECK(!throws_as<std::invalid_argument>([&] {
            PlaceController controller(valid, ik);
            (void)controller;
        }));
    }
}

bool begin_rejects(Fixture fixture, const PlaceCandidate& candidate) {
    refresh(fixture);
    PlaceController controller = make_controller(fixture);
    return !controller.begin({fixture.input, candidate}).accepted;
}

void test_begin_revalidates_complete_selection_and_is_atomic() {
    Fixture fixture = make_fixture();
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);

    const PlaceCandidate selected = begin.candidate;
    const auto forged_rejects = [&](const std::function<void(PlaceCandidate&)>& mutate) {
        PlaceCandidate forged = selected;
        mutate(forged);
        TEST_CHECK(begin_rejects(copy_fixture(fixture), forged));
    };
    forged_rejects([](PlaceCandidate& value) {
        value.mode = PlaceMotionMode::ReversedPickup;
    });
    forged_rejects([](PlaceCandidate& value) { value.selection_id ^= 1U; });
    forged_rejects([](PlaceCandidate& value) { ++value.source_id; });
    forged_rejects([](PlaceCandidate& value) { ++value.clip; });
    forged_rejects([](PlaceCandidate& value) { ++value.entry_frame; });
    forged_rejects([](PlaceCandidate& value) { ++value.commit_frame; });
    forged_rejects([](PlaceCandidate& value) { ++value.release_frame; });
    forged_rejects([](PlaceCandidate& value) { ++value.stop_frame; });
    forged_rejects([](PlaceCandidate& value) { value.direction = -value.direction; });
    forged_rejects([](PlaceCandidate& value) {
        value.scene_from_source.position.x += 0.001F;
    });
    forged_rejects([](PlaceCandidate& value) {
        value.scene_from_source.rotation = quat_from_angle_axis(
            0.001F, vec3(0.0F, 1.0F, 0.0F));
    });
    forged_rejects([](PlaceCandidate& value) {
        value.staging_root_world.position.z += 0.001F;
    });
    forged_rejects([](PlaceCandidate& value) {
        value.staging_root_world.rotation = quat_from_angle_axis(
            0.001F, vec3(0.0F, 1.0F, 0.0F));
    });
    forged_rejects([](PlaceCandidate& value) { value.entry_root_offset.x += 0.001F; });
    forged_rejects([](PlaceCandidate& value) { value.entry_yaw_offset += 0.001F; });
    forged_rejects([](PlaceCandidate& value) { value.total_cost += 0.001F; });
    forged_rejects([](PlaceCandidate& value) {
        value.timing.canonical_fps = 24.0F;
    });
    forged_rejects([](PlaceCandidate& value) { value.timing.playback_speed = 0.9F; });
    forged_rejects([](PlaceCandidate& value) {
        value.timing.entry_blend_seconds = 0.20F;
    });
    forged_rejects([](PlaceCandidate& value) {
        value.timing.reversed_commit_seconds = 0.40F;
    });
    forged_rejects([](PlaceCandidate& value) {
        value.timing.maximum_alignment_seconds = 0.90F;
    });
    forged_rejects([](PlaceCandidate& value) {
        value.match.maximum_entry_root_error_m = 0.2F;
    });
    forged_rejects([](PlaceCandidate& value) {
        value.match.maximum_entry_yaw_error_radians = 0.30F;
    });
    forged_rejects([](PlaceCandidate& value) { value.ik.damping += 0.001F; });

    const auto changed_input_rejects = [&](const std::function<void(Fixture&)>& mutate) {
        Fixture changed = copy_fixture(fixture);
        mutate(changed);
        PlaceController changed_controller = make_controller(fixture);
        TEST_CHECK(!changed_controller.begin({changed.input, selected}).accepted);
    };
    changed_input_rejects([](Fixture& value) { ++value.input.held_target.generation; });
    changed_input_rejects([](Fixture& value) {
        value.input.current_pose.positions[kRoot].x += 0.001F;
    });
    changed_input_rejects([](Fixture& value) {
        value.input.current_object_world.position.y += 0.001F;
    });
    changed_input_rejects([](Fixture& value) {
        value.input.surface.surface_world.position.x += 0.001F;
    });
    changed_input_rejects([](Fixture& value) { value.input.pickup_database = nullptr; });
    changed_input_rejects([](Fixture& value) { value.input.library = nullptr; });

    {
        Fixture stale = copy_fixture(fixture);
        PlaceBeginInput stale_begin = selected_begin(stale);
        PlaceController stale_controller = make_controller(stale);
        stale.library.recorded.front().poses.front().velocities[kRoot].x +=
            0.001F;
        TEST_CHECK(!stale_controller.begin(stale_begin).accepted);
        TEST_CHECK(stale_controller.begin(selected_begin(stale)).accepted);
    }

    Fixture retry = copy_fixture(fixture);
    PlaceBeginInput retry_begin = selected_begin(retry);
    PlaceController retry_controller = make_controller(retry);
    PlaceCandidate bad = retry_begin.candidate;
    bad.selection_id ^= 1U;
    TEST_CHECK(!retry_controller.begin({retry.input, bad}).accepted);
    TEST_CHECK(retry_controller.begin(retry_begin).accepted);
}

void test_begin_rejects_constructor_ik_identity_mismatch_atomically() {
    using Member = float IKConfig::*;
    const std::array<Member, 8> fields = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (size_t index = 0; index < fields.size(); ++index) {
        Fixture fixture = make_fixture();
        const PlaceBeginInput original = selected_begin(fixture);
        IKConfig constructor_ik = fixture.input.ik;
        const float direction = index < 2U
            ? 0.0F
            : std::numeric_limits<float>::infinity();
        constructor_ik.*fields[index] = std::nextafter(
            constructor_ik.*fields[index], direction);
        PlaceController controller(
            controller_config(fixture), constructor_ik);
        TEST_CHECK(!controller.begin(original).accepted);

        fixture.input.ik = constructor_ik;
        TEST_CHECK(controller.begin(selected_begin(fixture)).accepted);
    }

    Fixture fixture = make_fixture();
    const PlaceBeginInput original = selected_begin(fixture);
    IKConfig constructor_ik = fixture.input.ik;
    ++constructor_ik.maximum_iterations;
    PlaceController controller(controller_config(fixture), constructor_ik);
    TEST_CHECK(!controller.begin(original).accepted);
    fixture.input.ik = constructor_ik;
    TEST_CHECK(controller.begin(selected_begin(fixture)).accepted);
}

void test_begin_rejects_every_ik_identity_mismatch() {
    using Member = float IKConfig::*;
    const std::array<Member, 8> fields = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (Member member : fields) {
        Fixture fixture = make_fixture();
        PlaceBeginInput begin = selected_begin(fixture);
        begin.match_input.ik.*member = std::nextafter(
            begin.match_input.ik.*member,
            std::numeric_limits<float>::infinity());
        PlaceController controller = make_controller(fixture);
        TEST_CHECK(!controller.begin(begin).accepted);

        fixture = make_fixture();
        begin = selected_begin(fixture);
        begin.candidate.ik.*member = std::nextafter(
            begin.candidate.ik.*member,
            std::numeric_limits<float>::infinity());
        controller = make_controller(fixture);
        TEST_CHECK(!controller.begin(begin).accepted);
    }
    {
        Fixture fixture = make_fixture();
        PlaceBeginInput begin = selected_begin(fixture);
        ++begin.match_input.ik.maximum_iterations;
        PlaceController controller = make_controller(fixture);
        TEST_CHECK(!controller.begin(begin).accepted);
        begin = selected_begin(fixture);
        ++begin.candidate.ik.maximum_iterations;
        controller = make_controller(fixture);
        TEST_CHECK(!controller.begin(begin).accepted);
    }
}

void test_success_is_hand_derived_one_shot_and_frozen_after_ack() {
    Fixture fixture = make_fixture();
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);

    int release_pulses = 0;
    PlaceStep release{};
    for (int tick = 0; tick < 64; ++tick) {
        const PlaceStep step = controller.update(0.04F);
        TEST_CHECK(near(
            step.object_world,
            hand_derived_object(step.pose, fixture.input.held_affordance),
            2.0e-4F));
        if (step.release_due) {
            ++release_pulses;
            release = step;
            break;
        }
        TEST_CHECK(!step.retract_finished);
        TEST_CHECK(!step.recover_to_carry);
    }
    TEST_CHECK(release_pulses == 1);
    TEST_CHECK(release.release_due);
    TEST_CHECK(release.committed);
    TEST_CHECK(release.phase == PlacePhase::Release);
    TEST_CHECK(release.hand_position_error_m <= 0.02F);
    TEST_CHECK(release.hand_orientation_error_radians <=
               kReleaseOrientationCap);
    TEST_CHECK(release.actual_fit.accepted);
    TEST_CHECK(release.support_sweep_clear);

    const PlaceStep pending = controller.update(0.04F);
    TEST_CHECK(!pending.release_due);
    TEST_CHECK(near(pending.pose, release.pose));
    TEST_CHECK(near(pending.object_world, release.object_world));
    TEST_CHECK(near(
        pending.object_world,
        hand_derived_object(pending.pose, fixture.input.held_affordance),
        2.0e-4F));

    controller.acknowledge_release(release.object_world);
    bool finished = false;
    for (int tick = 0; tick < 32; ++tick) {
        const PlaceStep step = controller.update(0.04F);
        TEST_CHECK(!step.release_due);
        TEST_CHECK(near(step.object_world, release.object_world));
        if (step.retract_finished) {
            TEST_CHECK(step.phase == PlacePhase::Finished);
            finished = true;
            break;
        }
    }
    TEST_CHECK(finished);
}

void test_lifecycle_guards_and_acknowledgement_are_atomic() {
    Fixture fixture = make_fixture();
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(throws_as<std::logic_error>([&] {
        (void)controller.update(0.04F);
    }));
    TEST_CHECK(throws_as<std::logic_error>([&] {
        (void)controller.cancel();
    }));
    TEST_CHECK(throws_as<std::logic_error>([&] {
        controller.acknowledge_release(Transform{});
    }));

    const PlaceBeginInput begin = selected_begin(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    TEST_CHECK(throws_as<std::logic_error>([&] {
        controller.acknowledge_release(fixture.input.current_object_world);
    }));

    const PlaceStep release = run_to_terminal_before_ack(controller);
    TEST_CHECK(release.release_due);
    Transform invalid = release.object_world;
    invalid.position.x = std::numeric_limits<float>::quiet_NaN();
    TEST_CHECK(throws_as<std::invalid_argument>([&] {
        controller.acknowledge_release(invalid);
    }));
    Transform mismatched = release.object_world;
    mismatched.position.z += 0.001F;
    TEST_CHECK(throws_as<std::invalid_argument>([&] {
        controller.acknowledge_release(mismatched);
    }));

    const PlaceStep still_pending = controller.update(0.04F);
    TEST_CHECK(!still_pending.release_due);
    TEST_CHECK(near(still_pending.pose, release.pose));
    TEST_CHECK(near(still_pending.object_world, release.object_world));

    Transform committed = release.object_world;
    committed.rotation = -committed.rotation;
    controller.acknowledge_release(committed);
    TEST_CHECK(throws_as<std::logic_error>([&] {
        controller.acknowledge_release(committed);
    }));
    const PlaceStep retract = controller.update(0.04F);
    TEST_CHECK(exactly_equal(retract.object_world, committed));
    TEST_CHECK(!retract.release_due);
}

void test_begin_reuses_only_terminal_controllers_atomically() {
    Fixture fixture = make_fixture();
    const PlaceBeginInput begin = selected_begin(fixture);

    {
        PlaceController trial = make_controller(fixture);
        PlaceController control = make_controller(fixture);
        TEST_CHECK(trial.begin(begin).accepted);
        TEST_CHECK(control.begin(begin).accepted);
        TEST_CHECK(!trial.begin(begin).accepted);
        TEST_CHECK(equivalent(
            trial.update(0.04F), control.update(0.04F)));
    }

    {
        PlaceController trial = make_controller(fixture);
        PlaceController control = make_controller(fixture);
        TEST_CHECK(trial.begin(begin).accepted);
        TEST_CHECK(control.begin(begin).accepted);
        PlaceStep trial_step{};
        PlaceStep control_step{};
        do {
            trial_step = trial.update(0.04F);
            control_step = control.update(0.04F);
            TEST_CHECK(equivalent(trial_step, control_step));
        } while (!trial_step.committed);
        TEST_CHECK(!trial.begin(begin).accepted);
        TEST_CHECK(equivalent(
            trial.update(0.04F), control.update(0.04F)));
    }

    {
        PlaceController trial = make_controller(fixture);
        PlaceController control = make_controller(fixture);
        TEST_CHECK(trial.begin(begin).accepted);
        TEST_CHECK(control.begin(begin).accepted);
        const PlaceStep trial_release = run_to_terminal_before_ack(trial);
        const PlaceStep control_release = run_to_terminal_before_ack(control);
        TEST_CHECK(equivalent(trial_release, control_release));
        TEST_CHECK(trial_release.release_due);
        TEST_CHECK(!trial.begin(begin).accepted);
        TEST_CHECK(equivalent(
            trial.update(0.04F), control.update(0.04F)));
    }

    {
        PlaceController controller = make_controller(fixture);
        TEST_CHECK(controller.begin(begin).accepted);
        for (int tick = 0; tick < 4; ++tick) {
            TEST_CHECK(!controller.update(0.04F).committed);
        }
        TEST_CHECK(controller.cancel().recover_to_carry);
        Fixture retry = make_fixture();
        ++retry.input.held_target.generation;
        TEST_CHECK(controller.begin(selected_begin(retry)).accepted);
    }

    {
        Fixture failed = make_fixture();
        failed.input.surface.affordances.front().object_in_surface.position.x =
            0.35F;
        failed.input.place_affordance =
            failed.input.surface.affordances.front();
        restage_fixture(failed);
        const PlaceBeginInput failed_begin = selected_begin(failed);
        PlaceController controller = make_controller(failed);
        TEST_CHECK(controller.begin(failed_begin).accepted);
        RecordedPlaceClip& clip = failed.library.recorded.front();
        clip.poses[static_cast<size_t>(clip.release_frame)]
            .positions[kRightHand].x += std::nextafter(0.02F, 0.0F);
        const PlaceStep recovery = run_to_terminal_before_ack(controller);
        TEST_CHECK(recovery.recover_to_carry);
        Fixture retry = make_fixture();
        ++retry.input.held_target.generation;
        TEST_CHECK(controller.begin(selected_begin(retry)).accepted);
    }

    {
        PlaceController controller = make_controller(fixture);
        TEST_CHECK(controller.begin(begin).accepted);
        const PlaceStep release = run_to_terminal_before_ack(controller);
        TEST_CHECK(release.release_due);
        controller.acknowledge_release(release.object_world);
        PlaceStep finished{};
        do {
            finished = controller.update(0.04F);
        } while (!finished.retract_finished);
        Fixture next = make_fixture();
        ++next.input.held_target.generation;
        const PlaceBeginInput next_begin = selected_begin(next);
        TEST_CHECK(controller.begin(next_begin).accepted);
        TEST_CHECK(!controller.update(0.04F).recover_to_carry);
    }
}

void test_root_yaw_warp_uses_release_smoothstep_and_seven_tick_entry_blend() {
    Fixture fixture = make_fixture();
    const Transform staging = root_transform(fixture.input.current_pose);
    const float yaw_offset = 0.30F;
    const Transform displaced{
        staging.position + vec3(0.20F, 0.0F, 0.0F),
        quat_normalize(quat_mul(
            quat_from_angle_axis(yaw_offset, vec3(0.0F, 1.0F, 0.0F)),
            staging.rotation)),
    };
    map_current_rigidly(fixture, displaced);
    PlaceBeginInput begin = selected_begin(fixture);
    TEST_CHECK(near(begin.candidate.entry_root_offset.x, 0.20F));
    TEST_CHECK(near(begin.candidate.entry_yaw_offset, yaw_offset));

    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    PlacePlayer oracle;
    oracle.start(begin.candidate, begin.match_input);
    for (int tick = 1; tick <= 7; ++tick) {
        oracle.advance(0.04F);
        const PlaceSample source = oracle.sample();
        const float raw_progress = static_cast<float>(source.source_frame) /
            static_cast<float>(begin.candidate.release_frame);
        const float progress = std::clamp(raw_progress, 0.0F, 1.0F);
        const float smoothed = progress * progress * (3.0F - 2.0F * progress);
        const float root_weight = 1.0F - smoothed;
        Pose corrected = source.pose;
        corrected.positions[kRoot] = corrected.positions[kRoot] +
            root_weight * begin.candidate.entry_root_offset;
        corrected.rotations[kRoot] = quat_normalize(quat_mul(
            quat_from_angle_axis(
                root_weight * begin.candidate.entry_yaw_offset,
                vec3(0.0F, 1.0F, 0.0F)),
            corrected.rotations[kRoot]));
        const float blend_raw = static_cast<float>(tick) / 7.0F;
        const float blend = blend_raw * blend_raw * (3.0F - 2.0F * blend_raw);
        const Pose expected = interpolate_pose(
            begin.match_input.current_pose, corrected, blend);
        const PlaceStep actual = controller.update(0.04F);
        TEST_CHECK(near(
            actual.pose.positions[kRoot],
            expected.positions[kRoot],
            2.0e-5F));
        TEST_CHECK(near(
            actual.pose.rotations[kRoot],
            expected.rotations[kRoot],
            2.0e-5F));
        if (tick == 6) {
            TEST_CHECK(!near(
                actual.pose.positions[kRoot],
                corrected.positions[kRoot],
                1.0e-6F));
        }
        if (tick == 7) {
            TEST_CHECK(near(
                actual.pose.positions[kRoot],
                corrected.positions[kRoot],
                1.0e-6F));
        }
    }
}

void test_fractional_speed_weights_follow_exact_source_position() {
    Fixture fixture = make_fixture();
    fixture.input.timing.playback_speed = 0.85F;
    const Transform staging = root_transform(fixture.input.current_pose);
    map_current_rigidly(
        fixture,
        Transform{
            staging.position + vec3(0.20F, 0.0F, 0.0F),
            staging.rotation});
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    PlacePlayer oracle;
    oracle.start(begin.candidate, begin.match_input);
    float previous_weight = 1.0F;
    for (int tick = 1; tick <= 5; ++tick) {
        oracle.advance(0.04F);
        const PlaceSample source = oracle.sample();
        const double exact_source = oracle.source_frame_exact();
        const double denominator =
            begin.candidate.release_frame - begin.candidate.entry_frame;
        const float progress = static_cast<float>(
            (exact_source - begin.candidate.entry_frame) / denominator);
        const float smoothed = progress * progress * (3.0F - 2.0F * progress);
        const float weight = 1.0F - smoothed;
        TEST_CHECK(weight < previous_weight);
        previous_weight = weight;
        Pose corrected = source.pose;
        corrected.positions[kRoot] = corrected.positions[kRoot] +
            weight * begin.candidate.entry_root_offset;
        const float blend_raw = static_cast<float>(tick) / 7.0F;
        const float blend = blend_raw * blend_raw * (3.0F - 2.0F * blend_raw);
        const Pose expected = interpolate_pose(
            begin.match_input.current_pose, corrected, blend);
        const PlaceStep actual = controller.update(0.04F);
        TEST_CHECK(near(
            actual.pose.positions[kRoot],
            expected.positions[kRoot],
            2.0e-5F));
    }
}

void test_step_reports_exact_source_and_zeroes_corrections_after_release() {
    Fixture fixture = make_fixture();
    const Transform staging = root_transform(fixture.input.current_pose);
    map_current_rigidly(
        fixture,
        Transform{
            staging.position + vec3(0.10F, 0.0F, 0.0F),
            quat_from_angle_axis(0.10F, vec3(0.0F, 1.0F, 0.0F))});
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    PlacePlayer oracle;
    oracle.start(begin.candidate, begin.match_input);
    oracle.advance(0.04F);
    const PlaceStep first = controller.update(0.04F);
    TEST_CHECK(first.source_frame_exact == oracle.source_frame_exact());
    TEST_CHECK(first.requested_root_correction_m > 0.0F);
    TEST_CHECK(first.applied_root_correction_m >= 0.0F);
    TEST_CHECK(first.requested_yaw_correction_radians > 0.0F);
    TEST_CHECK(first.applied_yaw_correction_radians >= 0.0F);
    TEST_CHECK(first.requested_hand_correction_m >= 0.0F);
    TEST_CHECK(first.applied_hand_correction_m >= 0.0F);
    TEST_CHECK(first.requested_hand_orientation_radians >= 0.0F);
    TEST_CHECK(first.applied_hand_orientation_radians >= 0.0F);

    const PlaceStep release = run_to_terminal_before_ack(controller);
    TEST_CHECK(release.release_due);
    controller.acknowledge_release(release.object_world);
    const PlaceStep retract = controller.update(0.04F);
    TEST_CHECK(retract.source_frame_exact >= 0.0);
    TEST_CHECK(retract.requested_root_correction_m == 0.0F);
    TEST_CHECK(retract.applied_root_correction_m == 0.0F);
    TEST_CHECK(retract.requested_yaw_correction_radians == 0.0F);
    TEST_CHECK(retract.applied_yaw_correction_radians == 0.0F);
    TEST_CHECK(retract.requested_hand_correction_m == 0.0F);
    TEST_CHECK(retract.applied_hand_correction_m == 0.0F);
    TEST_CHECK(retract.requested_hand_orientation_radians == 0.0F);
    TEST_CHECK(retract.applied_hand_orientation_radians == 0.0F);
}

void test_authoritative_ik_position_limit_exact_and_nextabove() {
    Fixture fixture = make_fixture();
    const float limit = 0.02F;
    set_recorded_grasp(
        fixture.library.recorded.front(),
        Transform{vec3(limit, 0.0F, 0.0F), quat()});
    fixture.input.ik.maximum_request_position_m = limit;
    fixture.input.ik.accepted_position_m = limit;
    fixture.input.match.maximum_entry_root_error_m = 0.01F;
    fixture.input.match.maximum_entry_yaw_error_radians = 0.01F;
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    const PlaceStep exact = run_to_terminal_before_ack(controller);
    TEST_CHECK(exact.release_due);
    TEST_CHECK(!exact.recover_to_carry);

    fixture = make_fixture();
    set_recorded_grasp(
        fixture.library.recorded.front(),
        Transform{vec3(limit, 0.0F, 0.0F), quat()});
    fixture.input.ik.maximum_request_position_m = limit;
    fixture.input.ik.accepted_position_m = limit;
    begin = selected_begin(fixture);
    controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    move_release_hand_to_first_position_above(
        fixture.library.recorded.front(),
        begin.candidate,
        compose(
            placement_goal_world(
                fixture.input.surface,
                fixture.input.place_affordance.object_in_surface),
            fixture.input.held_affordance.hand_in_object),
        limit);
    const PlaceStep above = run_to_terminal_before_ack(controller);
    TEST_CHECK(above.recover_to_carry);
    TEST_CHECK(!above.release_due);
    TEST_CHECK(above.reason == Reason::CorrectionLimit);
}

void test_authoritative_ik_orientation_limit_exact_and_nextabove() {
    Fixture fixture = make_fixture();
    const float limit = kReleaseOrientationCap;
    set_recorded_grasp(
        fixture.library.recorded.front(),
        Transform{
            vec3(),
            quat_from_angle_axis(limit, vec3(0.0F, 1.0F, 0.0F))});
    fixture.input.ik.maximum_request_orientation_radians = limit;
    fixture.input.ik.accepted_orientation_radians = limit;
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    const PlaceStep exact = run_to_terminal_before_ack(controller);
    TEST_CHECK(exact.release_due);
    TEST_CHECK(!exact.recover_to_carry);

    fixture = make_fixture();
    set_recorded_grasp(
        fixture.library.recorded.front(),
        Transform{
            vec3(),
            quat_from_angle_axis(limit, vec3(0.0F, 1.0F, 0.0F))});
    fixture.input.ik.maximum_request_orientation_radians = limit;
    fixture.input.ik.accepted_orientation_radians = limit;
    begin = selected_begin(fixture);
    controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    move_release_hand_to_first_orientation_above(
        fixture.library.recorded.front(),
        begin.candidate,
        compose(
            placement_goal_world(
                fixture.input.surface,
                fixture.input.place_affordance.object_in_surface),
            fixture.input.held_affordance.hand_in_object),
        limit);
    const PlaceStep above = run_to_terminal_before_ack(controller);
    TEST_CHECK(above.recover_to_carry);
    TEST_CHECK(!above.release_due);
    TEST_CHECK(above.reason == Reason::CorrectionLimit);
}

void test_far_authored_entry_uses_only_mapped_release_residual() {
    Fixture fixture = make_fixture();
    RecordedPlaceClip& clip = fixture.library.recorded.front();
    for (int32_t frame = clip.entry_frame; frame <= 6; ++frame) {
        clip.object_poses[static_cast<size_t>(frame)].position.x += 2.0F;
        clip.poses[static_cast<size_t>(frame)].positions[kRightHand].x += 2.0F;
    }
    stage_from_recorded_entry(fixture);
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    const PlaceStep terminal = run_to_terminal_before_ack(controller);
    TEST_CHECK(terminal.release_due);
    TEST_CHECK(!terminal.recover_to_carry);
}

void test_controller_does_not_duplicate_dynamic_readiness_gate() {
    Fixture fixture = make_fixture();
    fixture.input.match.maximum_entry_root_error_m = 0.01F;
    fixture.input.match.maximum_entry_yaw_error_radians = 0.01F;
    const Transform current = root_transform(fixture.input.current_pose);
    map_current_rigidly(
        fixture,
        Transform{
            current.position + vec3(0.50F, 0.0F, 0.0F),
            quat_normalize(quat_mul(
                quat_from_angle_axis(0.30F, vec3(0.0F, 1.0F, 0.0F)),
                current.rotation)),
        });
    const PlaceStagingPreview preview = preview_place_motion(fixture.input);
    TEST_CHECK(preview.accepted);
    TEST_CHECK(!preview.ready);
    TEST_CHECK(preview.root_error_m > 0.25F);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin({fixture.input, preview.candidate}).accepted);
    const PlaceStep first = controller.update(0.04F);
    TEST_CHECK(!first.recover_to_carry);
}

void test_release_error_boundaries_are_distinct_from_ik_limits() {
    const auto goal_hand = [](const Fixture& fixture) {
        return compose(
            placement_goal_world(
                fixture.input.surface,
                fixture.input.place_affordance.object_in_surface),
            fixture.input.held_affordance.hand_in_object);
    };

    for (bool above : {false, true}) {
        Fixture fixture = make_fixture();
        PlaceBeginInput begin = selected_begin(fixture);
        PlaceController controller = make_controller(fixture);
        TEST_CHECK(controller.begin(begin).accepted);
        set_release_position_error_boundary(
            fixture.library.recorded.front(),
            begin.candidate,
            goal_hand(fixture),
            0.02F,
            above);
        const PlaceStep terminal = run_to_terminal_before_ack(controller);
        if (above) {
            TEST_CHECK(terminal.recover_to_carry);
            TEST_CHECK(!terminal.release_due);
            TEST_CHECK(terminal.reason == Reason::ReleasePosition);
            TEST_CHECK(terminal.hand_position_error_m > 0.02F);
        } else {
            TEST_CHECK(terminal.release_due);
            TEST_CHECK(!terminal.recover_to_carry);
            TEST_CHECK(terminal.hand_position_error_m <= 0.02F);
        }
    }

    for (bool above : {false, true}) {
        Fixture fixture = make_fixture();
        PlaceBeginInput begin = selected_begin(fixture);
        PlaceController controller = make_controller(fixture);
        TEST_CHECK(controller.begin(begin).accepted);
        set_release_orientation_error_boundary(
            fixture.library.recorded.front(),
            begin.candidate,
            goal_hand(fixture),
            kReleaseOrientationCap,
            above);
        const PlaceStep terminal = run_to_terminal_before_ack(controller);
        if (above) {
            TEST_CHECK(terminal.recover_to_carry);
            TEST_CHECK(!terminal.release_due);
            TEST_CHECK(terminal.reason == Reason::ReleaseOrientation);
            TEST_CHECK(
                terminal.hand_orientation_error_radians >
                kReleaseOrientationCap);
        } else {
            TEST_CHECK(terminal.release_due);
            TEST_CHECK(!terminal.recover_to_carry);
            TEST_CHECK(
                terminal.hand_orientation_error_radians <=
                kReleaseOrientationCap);
        }
    }
}

void test_reverse_lifecycle_and_cancel_boundaries() {
    Fixture fixture = make_fixture(false);
    PlaceBeginInput begin = selected_begin(fixture);
    TEST_CHECK(begin.candidate.mode == PlaceMotionMode::ReversedPickup);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    PlaceStep release = run_to_terminal_before_ack(controller);
    TEST_CHECK(release.release_due);
    TEST_CHECK(!release.recover_to_carry);
    TEST_CHECK(release.source_frame == begin.candidate.release_frame);
    TEST_CHECK(near(
        release.object_world,
        hand_derived_object(release.pose, fixture.input.held_affordance),
        2.0e-4F));
    const PlaceStep pending = controller.update(0.04F);
    TEST_CHECK(!pending.release_due);
    TEST_CHECK(near(pending.object_world, release.object_world));
    controller.acknowledge_release(release.object_world);
    bool finished = false;
    for (int tick = 0; tick < 64; ++tick) {
        const PlaceStep retract = controller.update(0.04F);
        TEST_CHECK(near(retract.object_world, release.object_world));
        if (retract.retract_finished) {
            finished = true;
            break;
        }
    }
    TEST_CHECK(finished);

    fixture = make_fixture(false);
    begin = selected_begin(fixture);
    controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    PlaceStep precommit{};
    do {
        precommit = controller.update(0.04F);
    } while (!precommit.committed &&
             precommit.source_frame != begin.candidate.commit_frame + 1);
    TEST_CHECK(!precommit.committed);
    TEST_CHECK(precommit.phase == PlacePhase::Align);
    const PlaceStep cancelled = controller.cancel();
    TEST_CHECK(cancelled.recover_to_carry);
    TEST_CHECK(cancelled.reason == Reason::Cancelled);
    TEST_CHECK(near(cancelled.pose, precommit.pose));
    TEST_CHECK(near(cancelled.object_world, precommit.object_world));

    fixture = make_fixture(false);
    begin = selected_begin(fixture);
    controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    PlaceStep committed{};
    do {
        committed = controller.update(0.04F);
    } while (!committed.committed);
    TEST_CHECK(committed.source_frame == begin.candidate.commit_frame);
    TEST_CHECK(committed.phase == PlacePhase::Lower);
    const PlaceStep ignored = controller.cancel();
    TEST_CHECK(!ignored.recover_to_carry);
    TEST_CHECK(ignored.committed);
}

void test_exact_dt_rejection_is_atomic() {
    Fixture fixture = make_fixture();
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController trial = make_controller(fixture);
    PlaceController control = make_controller(fixture);
    TEST_CHECK(trial.begin(begin).accepted);
    TEST_CHECK(control.begin(begin).accepted);
    const float infinity = std::numeric_limits<float>::infinity();
    for (float dt : {
             0.0F,
             1.0F / 60.0F,
             std::numeric_limits<float>::quiet_NaN(),
             std::nextafter(0.04F, 0.0F),
             std::nextafter(0.04F, infinity)}) {
        TEST_CHECK(throws_as<std::invalid_argument>([&] {
            (void)trial.update(dt);
        }));
    }
    const PlaceStep actual = trial.update(0.04F);
    const PlaceStep expected = control.update(0.04F);
    TEST_CHECK(actual.source_frame == expected.source_frame);
    TEST_CHECK(actual.phase == expected.phase);
    TEST_CHECK(near(actual.pose, expected.pose));
    TEST_CHECK(near(actual.object_world, expected.object_world));
}

void test_cancel_is_exactly_precommit_and_recovers_last_safe_pair() {
    Fixture fixture = make_fixture();
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    PlaceStep last{};
    int align_ticks = 0;
    while (true) {
        const PlaceStep step = controller.update(0.04F);
        if (step.committed) break;
        last = step;
        ++align_ticks;
        if (step.source_frame == begin.candidate.commit_frame - 1) break;
    }
    TEST_CHECK(align_ticks >= 4);
    TEST_CHECK(last.phase == PlacePhase::Align);
    TEST_CHECK(!last.committed);
    const PlaceStep cancelled = controller.cancel();
    TEST_CHECK(cancelled.recover_to_carry);
    TEST_CHECK(cancelled.reason == Reason::Cancelled);
    TEST_CHECK(near(cancelled.pose, last.pose));
    TEST_CHECK(near(cancelled.object_world, last.object_world));

    fixture = make_fixture();
    begin = selected_begin(fixture);
    PlaceController committed_controller = make_controller(fixture);
    TEST_CHECK(committed_controller.begin(begin).accepted);
    PlaceStep committed{};
    do {
        committed = committed_controller.update(0.04F);
    } while (!committed.committed);
    TEST_CHECK(committed.source_frame == begin.candidate.commit_frame);
    TEST_CHECK(committed.phase == PlacePhase::Lower);
    const PlaceStep ignored_at_commit = committed_controller.cancel();
    TEST_CHECK(!ignored_at_commit.recover_to_carry);
    TEST_CHECK(ignored_at_commit.committed);
    const PlaceStep after = committed_controller.update(0.04F);
    const PlaceStep ignored_after = committed_controller.cancel();
    TEST_CHECK(!ignored_after.recover_to_carry);
    TEST_CHECK(ignored_after.committed);
    TEST_CHECK(ignored_after.source_frame == after.source_frame);
}

void shift_recorded_hand(RecordedPlaceClip& clip, int32_t frame, vec3 delta) {
    clip.poses.at(static_cast<size_t>(frame)).positions[kRightHand] =
        clip.poses.at(static_cast<size_t>(frame)).positions[kRightHand] + delta;
}

PlaceStep run_to_terminal_before_ack(PlaceController& controller) {
    PlaceStep step{};
    for (int tick = 0; tick < 64; ++tick) {
        step = controller.update(0.04F);
        if (step.release_due || step.recover_to_carry) return step;
    }
    TEST_CHECK(false);
    return step;
}

void test_actual_release_footprint_failure_recovers_attached() {
    Fixture fixture = make_fixture();
    fixture.input.surface.affordances.front().object_in_surface.position.x = 0.35F;
    fixture.input.place_affordance = fixture.input.surface.affordances.front();
    restage_fixture(fixture);
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);

    shift_recorded_hand(
        fixture.library.recorded.front(),
        fixture.library.recorded.front().release_frame,
        vec3(std::nextafter(0.02F, 0.0F), 0.0F, 0.0F));
    PlaceStep last_safe{};
    PlaceStep terminal{};
    for (int tick = 0; tick < 64; ++tick) {
        const PlaceStep step = controller.update(0.04F);
        if (step.recover_to_carry) {
            terminal = step;
            break;
        }
        last_safe = step;
    }
    TEST_CHECK(terminal.recover_to_carry);
    TEST_CHECK(!terminal.release_due);
    TEST_CHECK(terminal.reason == Reason::PlacementOutOfBounds);
    TEST_CHECK(near(terminal.pose, last_safe.pose));
    TEST_CHECK(near(terminal.object_world, last_safe.object_world));
    const PlaceStep repeated = controller.update(0.04F);
    TEST_CHECK(repeated.recover_to_carry);
    TEST_CHECK(!repeated.release_due);
}

void test_actual_support_gap_lowest_and_overhead_failures() {
    const auto run_case = [](std::string_view name,
                             const std::function<void(Fixture&)>& author,
                             vec3 release_delta) {
        Fixture fixture = make_fixture();
        author(fixture);
        restage_fixture(fixture);
        PlaceBeginInput begin = selected_begin(fixture);
        PlaceController controller = make_controller(fixture);
        TEST_CHECK(controller.begin(begin).accepted);
        shift_recorded_hand(
            fixture.library.recorded.front(),
            fixture.library.recorded.front().release_frame,
            release_delta);
        const PlaceStep terminal = run_to_terminal_before_ack(controller);
        if (!terminal.recover_to_carry) {
            std::cerr << "release geometry case did not recover: " << name
                      << " due=" << terminal.release_due
                      << " gap=" << terminal.actual_fit.support_gap_m
                      << " low=" << terminal.actual_fit.lowest_corner_m
                      << " high=" << terminal.actual_fit.highest_corner_m
                      << '\n';
        }
        TEST_CHECK(terminal.recover_to_carry);
        TEST_CHECK(!terminal.release_due);
        TEST_CHECK(terminal.reason == Reason::PlacementOutOfBounds);
        TEST_CHECK(terminal.hand_position_error_m <= 0.02F);
        if (name == "support gap") {
            TEST_CHECK(terminal.actual_fit.support_gap_m > 0.02F);
        } else if (name == "lowest") {
            TEST_CHECK(terminal.actual_fit.lowest_corner_m < -0.005F);
        } else if (name == "overhead") {
            TEST_CHECK(
                terminal.actual_fit.highest_corner_m >
                fixture.input.surface.overhead_clearance_m);
        }
    };

    run_case(
        "support gap",
        [](Fixture& fixture) {
            fixture.input.surface.affordances.front().object_in_surface.position.y =
                0.12F;
            fixture.input.place_affordance = fixture.input.surface.affordances.front();
        },
        vec3(0.0F, 0.021F, 0.0F));
    run_case("lowest", [](Fixture&) {}, vec3(0.0F, -0.006F, 0.0F));
    run_case(
        "overhead",
        [](Fixture& fixture) {
            fixture.input.surface.overhead_clearance_m = 0.20F;
        },
        vec3(0.0F, 0.001F, 0.0F));
}

void test_nonrelease_endpoint_and_swept_support_crossing_are_blocked() {
    Fixture fixture = make_fixture();
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    RecordedPlaceClip& clip = fixture.library.recorded.front();
    const int32_t crossing_frame = 2;
    const vec3 root = clip.poses[static_cast<size_t>(crossing_frame)].positions[kRoot];
    clip.poses[static_cast<size_t>(crossing_frame)].positions[kRightHand] =
        vec3(0.0F, -0.11F, 0.0F) - root;
    const PlaceStep terminal = run_to_terminal_before_ack(controller);
    TEST_CHECK(terminal.recover_to_carry);
    TEST_CHECK(terminal.reason == Reason::BlockedPath);
    TEST_CHECK(!terminal.release_due);
}

void test_rotated_release_bounds_cannot_escape_footprint_gate() {
    Fixture fixture = make_fixture();
    fixture.input.held_object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.35F)};
    fixture.input.object_dimensions = vec3(0.08F, 0.20F, 0.70F);
    fixture.input.surface.affordances.front().object_in_surface.position.x =
        0.35F;
    fixture.input.place_affordance = fixture.input.surface.affordances.front();
    RecordedPlaceClip& clip = fixture.library.recorded.front();
    clip.object_bounds = fixture.input.held_object_bounds;
    restage_fixture(fixture);
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);
    clip.poses[static_cast<size_t>(clip.release_frame)]
        .rotations[kRightHand] = quat_from_angle_axis(
            0.157079633F, vec3(0.0F, 1.0F, 0.0F));
    const PlaceStep terminal = run_to_terminal_before_ack(controller);
    TEST_CHECK(terminal.recover_to_carry);
    TEST_CHECK(!terminal.release_due);
    TEST_CHECK(terminal.reason == Reason::PlacementOutOfBounds);
    TEST_CHECK(!terminal.actual_fit.footprint_valid);
}

void test_offcenter_rotating_sweep_uses_angular_inflation() {
    Fixture fixture = make_fixture();
    fixture.input.held_object_bounds = {
        vec3(0.20F, 0.0F, 0.0F),
        vec3(0.02F, 0.10F, 0.02F)};
    fixture.input.object_dimensions = vec3(0.04F, 0.20F, 0.04F);
    fixture.input.surface.affordances.front().object_in_surface.position.x =
        -0.20F;
    fixture.input.place_affordance = fixture.input.surface.affordances.front();
    RecordedPlaceClip& clip = fixture.library.recorded.front();
    clip.object_bounds = fixture.input.held_object_bounds;
    restage_fixture(fixture);
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    TEST_CHECK(controller.begin(begin).accepted);

    for (const auto& [frame, angle] :
         {std::pair<int32_t, float>{9, -0.523598776F},
          std::pair<int32_t, float>{10, 0.523598776F}}) {
        Pose& pose = clip.poses[static_cast<size_t>(frame)];
        const Transform object{
            vec3(0.0F, 0.81F, 0.0F),
            quat_from_angle_axis(angle, vec3(0.0F, 1.0F, 0.0F)),
        };
        pose.positions[kRightHand] = object.position - pose.positions[kRoot];
        pose.rotations[kRightHand] = object.rotation;
    }
    PlaceStep first{};
    do {
        first = controller.update(0.04F);
        TEST_CHECK(!first.recover_to_carry);
    } while (first.source_frame < 9);
    TEST_CHECK(!first.recover_to_carry);
    const PlaceStep second = controller.update(0.04F);
    TEST_CHECK(second.recover_to_carry);
    TEST_CHECK(second.reason == Reason::BlockedPath);
    TEST_CHECK(!second.release_due);
    TEST_CHECK(near(second.pose, first.pose));
    TEST_CHECK(near(second.object_world, first.object_world));
}

void test_commit_time_is_bounded_at_every_playback_speed() {
    constexpr float strict_boundary_speed = 1.1428570747375488F;
    TEST_CHECK(
        strict_boundary_speed == std::nextafter(8.0F / 7.0F, 0.0F));
    for (bool recorded : {true, false}) {
        for (float speed : {
                 0.85F, 1.0F, strict_boundary_speed, 1.15F}) {
            Fixture fixture = make_fixture(recorded);
            fixture.input.timing.playback_speed = speed;
            PlaceBeginInput begin = selected_begin(fixture);
            const int64_t source_delta = std::abs(
                static_cast<int64_t>(begin.candidate.commit_frame) -
                begin.candidate.entry_frame);
            const int32_t expected_ticks = static_cast<int32_t>(std::ceil(
                static_cast<double>(source_delta) /
                static_cast<double>(speed)));
            PlaceController controller = make_controller(fixture);
            TEST_CHECK(controller.begin(begin).accepted);
            int ticks = 0;
            PlaceStep step{};
            do {
                step = controller.update(0.04F);
                ++ticks;
                TEST_CHECK(ticks < 64);
                if (ticks < expected_ticks) {
                    TEST_CHECK(!step.committed);
                }
            } while (!step.committed && !step.recover_to_carry);
            const float elapsed = static_cast<float>(ticks) * 0.04F;
            TEST_CHECK(step.committed);
            TEST_CHECK(ticks == expected_ticks);
            TEST_CHECK(step.source_frame == begin.candidate.commit_frame);
            TEST_CHECK(elapsed >= fixture.input.timing.entry_blend_seconds);
            TEST_CHECK(elapsed <= fixture.input.timing.maximum_alignment_seconds);
        }
    }
}

}  // namespace

int main() {
    test_frozen_public_contract();
    test_rotation_gate_is_scale_sign_and_boundary_stable();
    test_constructor_validates_every_configuration_family();
    test_begin_revalidates_complete_selection_and_is_atomic();
    test_begin_rejects_constructor_ik_identity_mismatch_atomically();
    test_begin_rejects_every_ik_identity_mismatch();
    test_success_is_hand_derived_one_shot_and_frozen_after_ack();
    test_lifecycle_guards_and_acknowledgement_are_atomic();
    test_begin_reuses_only_terminal_controllers_atomically();
    test_root_yaw_warp_uses_release_smoothstep_and_seven_tick_entry_blend();
    test_fractional_speed_weights_follow_exact_source_position();
    test_step_reports_exact_source_and_zeroes_corrections_after_release();
    test_authoritative_ik_position_limit_exact_and_nextabove();
    test_authoritative_ik_orientation_limit_exact_and_nextabove();
    test_far_authored_entry_uses_only_mapped_release_residual();
    test_controller_does_not_duplicate_dynamic_readiness_gate();
    test_release_error_boundaries_are_distinct_from_ik_limits();
    test_reverse_lifecycle_and_cancel_boundaries();
    test_exact_dt_rejection_is_atomic();
    test_cancel_is_exactly_precommit_and_recovers_last_safe_pair();
    test_actual_release_footprint_failure_recovers_attached();
    test_actual_support_gap_lowest_and_overhead_failures();
    test_nonrelease_endpoint_and_swept_support_crossing_are_blocked();
    test_rotated_release_bounds_cannot_escape_footprint_gate();
    test_offcenter_rotating_sweep_uses_angular_inflation();
    test_commit_time_is_bounded_at_every_playback_speed();
    if (g_failures != 0) {
        std::cerr << g_failures << " place controller checks failed\n";
        return 1;
    }
    return 0;
}
