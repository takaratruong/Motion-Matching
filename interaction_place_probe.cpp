#include "interaction_controller_adapter.h"
#include "tests/cpp/pick_entry_oracle_roots.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr float kDt = 1.0F / 25.0F;
constexpr float kFlatWalkSpeedMps = 0.50F;
constexpr float kFlatTurnSpeedRadians = 1.50F;
constexpr float kPi = 3.14159265358979323846F;
constexpr int kMaximumPickupUpdates = 1200;
constexpr int kMaximumStagingUpdates = 400;
constexpr int kMaximumPlaceUpdates = 1200;
constexpr size_t kStableHoldSamples = 5U;
constexpr float kStablePositionToleranceM = 0.020F;

[[noreturn]] void fail(std::string message) {
    throw std::runtime_error(std::move(message));
}

void require(bool condition, std::string_view message) {
    if (!condition) fail(std::string(message));
}

std::string_view reason_name(interaction::Reason reason) {
    using interaction::Reason;
    switch (reason) {
    case Reason::None: return "None";
    case Reason::PackUnavailable: return "PackUnavailable";
    case Reason::TargetUnavailable: return "TargetUnavailable";
    case Reason::TargetChanged: return "TargetChanged";
    case Reason::OutOfRange: return "OutOfRange";
    case Reason::NoCandidate: return "NoCandidate";
    case Reason::PoorMatch: return "PoorMatch";
    case Reason::BlockedPath: return "BlockedPath";
    case Reason::CorrectionLimit: return "CorrectionLimit";
    case Reason::Cancelled: return "Cancelled";
    case Reason::ContactPosition: return "ContactPosition";
    case Reason::ContactOrientation: return "ContactOrientation";
    case Reason::JointLimit: return "JointLimit";
    case Reason::LostContact: return "LostContact";
    case Reason::ClipEnded: return "ClipEnded";
    case Reason::Reset: return "Reset";
    case Reason::SurfaceUnavailable: return "SurfaceUnavailable";
    case Reason::SurfaceChanged: return "SurfaceChanged";
    case Reason::PlacementOutOfBounds: return "PlacementOutOfBounds";
    case Reason::ReleasePosition: return "ReleasePosition";
    case Reason::ReleaseOrientation: return "ReleaseOrientation";
    }
    fail("invalid runtime reason");
}

vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 3U;
    return vec3(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U));
}

quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 4U;
    return quat(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U),
        values.at(offset + 3U));
}

interaction::Transform frame_transform(
    const std::vector<float>& positions,
    const std::vector<float>& rotations,
    size_t frame) {
    return {read_vec3(positions, frame), read_quat(rotations, frame)};
}

interaction::Transform root_transform(const interaction::Pose& pose) {
    const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    return {pose.positions[root], pose.rotations[root]};
}

size_t hand_bone(interaction::Hand hand) {
    return hand == interaction::Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
}

interaction::Transform hand_transform(
    const interaction::Pose& pose,
    interaction::Hand hand) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    const size_t bone = hand_bone(hand);
    return {world.positions[bone], quat_normalize(world.rotations[bone])};
}

interaction::Transform database_hand_in_object(
    const interaction::Database& database,
    int32_t frame,
    interaction::Hand hand) {
    const interaction::Transform object = frame_transform(
        database.object_positions,
        database.object_rotations,
        static_cast<size_t>(frame));
    return interaction::compose(
        interaction::inverse(object),
        hand_transform(interaction::pose_at_frame(database, frame), hand));
}

bool exact(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool exact(quat left, quat right) {
    return left.w == right.w && left.x == right.x &&
           left.y == right.y && left.z == right.z;
}

bool exact(
    const interaction::Transform& left,
    const interaction::Transform& right) {
    return exact(left.position, right.position) &&
           exact(left.rotation, right.rotation);
}

enum class PositionMatrixRoot : uint8_t { Reach, Plus };

struct PositionMatrixRow {
    const char* label = "baseline_reach";
    PositionMatrixRoot pickup_root = PositionMatrixRoot::Reach;
    vec3 pickup_translation{};
    float pickup_yaw_radians = 0.0F;
    vec3 destination_translation{};
    float destination_yaw_radians = 0.0F;
};

const std::array<PositionMatrixRow, 3> kPositionMatrixRows{{
    {"baseline_reach",
     PositionMatrixRoot::Reach,
     vec3(),
     0.0F,
     vec3(),
     0.0F},
    {"positive_plus",
     PositionMatrixRoot::Plus,
     vec3(0.60F, 0.0F, -0.40F),
     kPi / 6.0F,
     vec3(-0.40F, 0.0F, 0.30F),
     -kPi / 6.0F},
    {"negative_reach",
     PositionMatrixRoot::Reach,
     vec3(-0.60F, 0.0F, 0.40F),
     -kPi / 6.0F,
     vec3(0.40F, 0.0F, -0.30F),
     kPi / 6.0F},
}};

bool identity_scene_transform(vec3 translation, float yaw_radians_value) {
    return exact(translation, vec3()) && yaw_radians_value == 0.0F;
}

interaction::Transform scene_transform(
    vec3 translation,
    float yaw_radians_value) {
    return {
        translation,
        quat_from_angle_axis(
            yaw_radians_value, vec3(0.0F, 1.0F, 0.0F)),
    };
}

bool same_target_locals(
    const interaction::InteractionTarget& authored,
    const interaction::InteractionTarget& transformed) {
    if (transformed.handle != authored.handle ||
        !exact(transformed.table_size, authored.table_size) ||
        transformed.object_profile_id != authored.object_profile_id ||
        !exact(
            transformed.object_dimensions,
            authored.object_dimensions) ||
        !exact(
            transformed.object_bounds.center_object,
            authored.object_bounds.center_object) ||
        !exact(
            transformed.object_bounds.half_extents_object,
            authored.object_bounds.half_extents_object) ||
        transformed.state != authored.state ||
        transformed.owner_request != authored.owner_request ||
        transformed.affordances.size() != authored.affordances.size()) {
        return false;
    }
    for (size_t index = 0; index < authored.affordances.size(); ++index) {
        if (!interaction::same_authored_grasp_affordance(
                authored.affordances[index],
                transformed.affordances[index])) {
            return false;
        }
    }
    return true;
}

void require_target_locals_unchanged(
    const interaction::InteractionTarget& authored,
    const interaction::InteractionTarget& transformed) {
    require(
        same_target_locals(authored, transformed),
        "pickup scene transform changed target-local data");
}

void require_authored_slot_identity_mutations_are_observable(
    interaction::InteractionTarget& authored) {
    require(
        !authored.affordances.empty(),
        "slot identity fixture has no grasp affordance");
    authored.affordances.front().interaction_slots = {
        {3U, -0.41F, -0.22F, 1.10F},
        {9U, 0.18F, -0.39F, 0.20F},
    };

    interaction::InteractionTarget reordered = authored;
    std::swap(
        reordered.affordances.front().interaction_slots[0],
        reordered.affordances.front().interaction_slots[1]);
    require(
        !same_target_locals(authored, reordered),
        "pickup scene slot order mutation preserved local identity");

    interaction::InteractionTarget changed = authored;
    changed.affordances.front()
        .interaction_slots[0].root_x_object_m += 0.001F;
    require(
        !same_target_locals(authored, changed),
        "pickup scene slot value mutation preserved local identity");
}

interaction::InteractionTarget transform_pickup_scene(
    const interaction::InteractionTarget& authored,
    const PositionMatrixRow& row) {
    interaction::InteractionTarget transformed = authored;
    if (!identity_scene_transform(
            row.pickup_translation, row.pickup_yaw_radians)) {
        const interaction::Transform scene_from_authored = scene_transform(
            row.pickup_translation, row.pickup_yaw_radians);
        transformed.table_world = interaction::compose(
            scene_from_authored, authored.table_world);
        transformed.object_world = interaction::compose(
            scene_from_authored, authored.object_world);
    }
    require_target_locals_unchanged(authored, transformed);
    return transformed;
}

void require_destination_locals_unchanged(
    const interaction::PlacementSurface& authored,
    const interaction::PlacementSurface& transformed) {
    require(
        transformed.handle == authored.handle &&
            exact(
                transformed.support_volume_size,
                authored.support_volume_size) &&
            transformed.half_extent_x_m == authored.half_extent_x_m &&
            transformed.half_extent_z_m == authored.half_extent_z_m &&
            transformed.overhead_clearance_m ==
                authored.overhead_clearance_m &&
            transformed.affordances.size() == authored.affordances.size(),
        "destination scene transform changed surface-local data");
    for (size_t index = 0; index < authored.affordances.size(); ++index) {
        const interaction::PlaceAffordance& expected =
            authored.affordances[index];
        const interaction::PlaceAffordance& actual =
            transformed.affordances[index];
        require(
            actual.id == expected.id &&
                exact(actual.object_in_surface, expected.object_in_surface) &&
                exact(
                    actual.support_point_object,
                    expected.support_point_object) &&
                exact(
                    actual.approach_direction_surface,
                    expected.approach_direction_surface) &&
                actual.clearance_radius == expected.clearance_radius,
            "destination transform changed a local place affordance");
    }
}

interaction::PlacementSurface transform_destination_scene(
    const interaction::PlacementSurface& authored,
    const PositionMatrixRow& row) {
    interaction::PlacementSurface transformed = authored;
    if (!identity_scene_transform(
            row.destination_translation,
            row.destination_yaw_radians)) {
        const interaction::Transform scene_from_authored = scene_transform(
            row.destination_translation, row.destination_yaw_radians);
        transformed.surface_world = interaction::compose(
            scene_from_authored, authored.surface_world);
        transformed.support_volume_world = interaction::compose(
            scene_from_authored, authored.support_volume_world);
    }
    require_destination_locals_unchanged(authored, transformed);
    return transformed;
}

bool exact_placement_fit(
    const interaction::PlacementFit& left,
    const interaction::PlacementFit& right) {
    return left.accepted == right.accepted &&
        left.reason == right.reason &&
        left.support_gap_m == right.support_gap_m &&
        left.lowest_corner_m == right.lowest_corner_m &&
        left.highest_corner_m == right.highest_corner_m &&
        left.footprint_valid == right.footprint_valid &&
        left.overhead_valid == right.overhead_valid;
}

bool exact(
    const interaction::Pose& left,
    const interaction::Pose& right) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (!exact(left.positions[bone], right.positions[bone]) ||
            !exact(left.velocities[bone], right.velocities[bone]) ||
            !exact(left.rotations[bone], right.rotations[bone]) ||
            !exact(
                left.angular_velocities[bone],
                right.angular_velocities[bone])) {
            return false;
        }
    }
    return left.hand_dof == right.hand_dof &&
           left.hand_dof_velocities == right.hand_dof_velocities &&
           left.foot_contacts == right.foot_contacts;
}

bool exact(
    const interaction::IKConfig& left,
    const interaction::IKConfig& right) {
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

bool near(vec3 left, vec3 right, float tolerance = 1.0e-5F) {
    return length(left - right) <= tolerance;
}

bool near(
    const interaction::Transform& left,
    const interaction::Transform& right,
    float tolerance = 1.0e-5F) {
    return near(left.position, right.position, tolerance) &&
           std::abs(quat_dot(left.rotation, right.rotation)) >=
               1.0F - tolerance;
}

float planar_distance(vec3 left, vec3 right) {
    return std::hypot(right.x - left.x, right.z - left.z);
}

float yaw_radians(quat rotation) {
    rotation = quat_normalize(rotation);
    return std::atan2(
        2.0F * (rotation.w * rotation.y + rotation.x * rotation.z),
        1.0F - 2.0F *
            (rotation.y * rotation.y + rotation.z * rotation.z));
}

float shortest_angle(float value) {
    return std::remainder(value, 2.0F * kPi);
}

interaction::Transform planar_alignment(
    interaction::Transform source,
    interaction::Transform target) {
    const float yaw = shortest_angle(
        yaw_radians(target.rotation) - yaw_radians(source.rotation));
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

int32_t first_phase_frame(
    const interaction::Database& database,
    uint32_t clip,
    interaction::Phase phase) {
    const int32_t start = database.range_starts.at(clip);
    const int32_t stop = database.range_stops.at(clip);
    for (int32_t frame = start; frame < stop; ++frame) {
        if (database.phases.at(static_cast<size_t>(frame)) ==
            static_cast<uint8_t>(phase)) {
            return frame;
        }
    }
    fail("clip does not contain a required phase");
}

void validate_native_pack(
    const interaction::Database& database,
    const interaction::Features& features) {
    require(
        database.fps_numerator == 25U && database.fps_denominator == 1U,
        "placement probe requires native exact 25 Hz data");
    require(
        database.frame_count == features.frame_count,
        "feature/database frame count mismatch");
    require(database.clip_count > 0U, "placement probe pack has no clips");
    require(
        database.range_starts.size() == database.clip_count &&
            database.range_stops.size() == database.clip_count &&
            database.active_hands.size() == database.clip_count,
        "placement probe clip metadata is incomplete");
    require(
        database.source_frames.size() == database.frame_count,
        "placement probe source frames are incomplete");

    int32_t previous_stop = 0;
    for (uint32_t clip = 0; clip < database.clip_count; ++clip) {
        const int32_t start = database.range_starts.at(clip);
        const int32_t stop = database.range_stops.at(clip);
        require(
            start == previous_stop && start < stop &&
                stop <= static_cast<int32_t>(database.frame_count),
            "placement probe clip ranges are not contiguous");
        require(
            database.active_hands.at(clip) <= 1U,
            "placement probe active hand is invalid");
        for (int32_t frame = start + 1; frame < stop; ++frame) {
            require(
                database.source_frames.at(static_cast<size_t>(frame)) ==
                    database.source_frames.at(
                        static_cast<size_t>(frame - 1)) + 1,
                "placement probe source frames are not contiguous");
        }
        previous_stop = stop;
    }
    require(
        previous_stop == static_cast<int32_t>(database.frame_count),
        "placement probe clip ranges do not cover the database");
    static_cast<void>(first_phase_frame(
        database, 0U, interaction::Phase::Reach));
    static_cast<void>(first_phase_frame(
        database, 0U, interaction::Phase::Contact));
    static_cast<void>(first_phase_frame(
        database, 0U, interaction::Phase::Lift));
    static_cast<void>(first_phase_frame(
        database, 0U, interaction::Phase::Hold));
}

struct CertifiedFixtureFrames {
    int32_t contact = -1;
    int32_t hold = -1;
    int32_t reverse_start = -1;
    int32_t last_contact = -1;
};

CertifiedFixtureFrames make_first_hold_window_unstable(
    interaction::Database& database) {
    CertifiedFixtureFrames frames{};
    frames.contact = first_phase_frame(
        database, 0U, interaction::Phase::Contact);
    frames.hold = first_phase_frame(
        database, 0U, interaction::Phase::Hold);
    frames.reverse_start =
        frames.hold + static_cast<int32_t>(kStableHoldSamples);
    const int32_t stop = database.range_stops.at(0U);
    require(
        frames.reverse_start < stop,
        "placement fixture lacks the second Hold window");

    const interaction::Hand hand = database.active_hands.at(0U) == 0U
        ? interaction::Hand::Left
        : interaction::Hand::Right;
    const size_t contact_offset = static_cast<size_t>(hand);
    for (int32_t frame = frames.hold;
         frame <= frames.reverse_start;
         ++frame) {
        require(
            database.phases.at(static_cast<size_t>(frame)) ==
                static_cast<uint8_t>(interaction::Phase::Hold),
            "placement fixture second window is not all Hold");
        require(
            database.hand_contacts.at(
                static_cast<size_t>(frame) * 2U + contact_offset) == 1U,
            "placement fixture second window lost contact");
    }

    const size_t position =
        (static_cast<size_t>(frames.hold) * g1_skeleton::BoneCount +
         hand_bone(hand)) * 3U;
    database.positions.at(position) += 0.030F;
    const interaction::Transform first = database_hand_in_object(
        database, frames.hold, hand);
    const interaction::Transform second = database_hand_in_object(
        database, frames.hold + 1, hand);
    require(
        length(first.position - second.position) >
            kStablePositionToleranceM,
        "placement fixture first Hold window remained stable");

    for (int32_t frame = stop - 1; frame >= frames.contact; --frame) {
        if (database.hand_contacts.at(
                static_cast<size_t>(frame) * 2U + contact_offset) == 1U) {
            frames.last_contact = frame;
            break;
        }
    }
    require(
        frames.last_contact > frames.reverse_start,
        "placement fixture last contact overlaps certified prefix");
    return frames;
}

interaction::Pose mapped_pose(
    const interaction::Database& database,
    int32_t frame,
    interaction::Transform mapping) {
    interaction::Pose pose = interaction::pose_at_frame(database, frame);
    const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const interaction::Transform mapped_root = interaction::compose(
        mapping, root_transform(pose));
    pose.positions[root] = mapped_root.position;
    pose.rotations[root] = mapped_root.rotation;
    pose.velocities[root] = quat_mul_vec3(
        mapping.rotation, pose.velocities[root]);
    pose.angular_velocities[root] = quat_mul_vec3(
        mapping.rotation, pose.angular_velocities[root]);
    return pose;
}

interaction::LocomotionSnapshot initial_locomotion(
    const interaction::Database& database,
    const interaction::InteractionTarget& target) {
    const interaction::Transform source_table = frame_transform(
        database.table_positions, database.table_rotations, 0U);
    const interaction::Transform mapping = interaction::compose(
        target.table_world, interaction::inverse(source_table));
    const int32_t entry = first_phase_frame(
        database, 0U, interaction::Phase::Reach);
    interaction::LocomotionSnapshot locomotion{};
    locomotion.pose = mapped_pose(database, entry, mapping);
    constexpr std::array<int32_t, 3> kFutureOffsets = {8, 17, 25};
    const int32_t stop = database.range_stops.at(0U);
    for (size_t sample = 0; sample < kFutureOffsets.size(); ++sample) {
        const interaction::Pose future = mapped_pose(
            database,
            std::min(entry + kFutureOffsets[sample], stop - 1),
            mapping);
        locomotion.future_root_positions[sample] =
            future.positions[g1_skeleton::Simulation];
        locomotion.future_root_rotations[sample] =
            future.rotations[g1_skeleton::Simulation];
    }
    return locomotion;
}

interaction::PickEntryRoot selected_canonical_root(
    const interaction::Database& database,
    const interaction::InteractionTarget& target,
    PositionMatrixRoot selected) {
    const pick_entry_oracle::OracleRoots roots =
        pick_entry_oracle::make_oracle_roots(database, target);
    switch (selected) {
    case PositionMatrixRoot::Reach: return roots.reach;
    case PositionMatrixRoot::Plus: return roots.plus;
    }
    fail("position matrix row has an invalid root selector");
}

std::string_view position_matrix_root_name(PositionMatrixRoot selected) {
    switch (selected) {
    case PositionMatrixRoot::Reach: return "Reach";
    case PositionMatrixRoot::Plus: return "Plus";
    }
    fail("position matrix row has an invalid root selector");
}

void require_selected_initial_root(
    const interaction::Database& database,
    const interaction::InteractionTarget& target,
    const PositionMatrixRow& row,
    const interaction::LocomotionSnapshot& locomotion) {
    const interaction::PickEntryRoot selected = selected_canonical_root(
        database, target, row.pickup_root);
    const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const vec3 actual = locomotion.pose.positions[root];
    require(
        std::hypot(
            actual.x - selected.world_x,
            actual.z - selected.world_z) <= 1.0e-5F &&
            std::abs(shortest_angle(
                yaw_radians(locomotion.pose.rotations[root]) -
                selected.world_yaw_radians)) <= 1.0e-5F,
        std::string(row.label) +
            " initial live root does not match selected canonical " +
            std::string(position_matrix_root_name(row.pickup_root)) +
            " root");
}

interaction::LocomotionSnapshot initial_locomotion_for_row(
    const interaction::Database& database,
    const interaction::InteractionTarget& target,
    const PositionMatrixRow& row) {
    interaction::LocomotionSnapshot locomotion = initial_locomotion(
        database, target);
    if (row.pickup_root == PositionMatrixRoot::Reach) {
        return locomotion;
    }
    const interaction::runtime_detail::PickSnapshotMap mapped =
        interaction::runtime_detail::map_pick_entry_snapshot(
            locomotion,
            selected_canonical_root(database, target, row.pickup_root));
    require(
        mapped.accepted && mapped.reason == interaction::Reason::None,
        std::string(row.label) +
            " canonical root rejected production snapshot mapping");
    return mapped.snapshot;
}

vec3 step_planar_toward(vec3 current, vec3 target, float distance) {
    const vec3 delta(target.x - current.x, 0.0F, target.z - current.z);
    const float remaining = length(delta);
    if (remaining <= distance || remaining == 0.0F) {
        return vec3(target.x, current.y, target.z);
    }
    return current + (distance / remaining) * delta;
}

class DeterministicFlatLiveProvider {
public:
    explicit DeterministicFlatLiveProvider(
        interaction::LocomotionSnapshot initial)
        : snapshot_(std::move(initial)) {}

    const interaction::LocomotionSnapshot& snapshot() const {
        return snapshot_;
    }

    void advance_toward(
        const interaction::RuntimeOutput& actual,
        interaction::Transform staging_root) {
        snapshot_.pose = actual.pose;
        for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
            snapshot_.pose.velocities[bone] = vec3();
            snapshot_.pose.angular_velocities[bone] = vec3();
        }
        std::fill(
            snapshot_.pose.hand_dof_velocities.begin(),
            snapshot_.pose.hand_dof_velocities.end(),
            0.0F);
        const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
        const vec3 current = snapshot_.pose.positions[root];
        const vec3 next = step_planar_toward(
            current, staging_root.position, kFlatWalkSpeedMps * kDt);
        snapshot_.pose.positions[root] = next;
        snapshot_.pose.velocities[root] = (next - current) / kDt;

        const quat current_rotation = snapshot_.pose.rotations[root];
        const float yaw_error = std::abs(shortest_angle(
            yaw_radians(staging_root.rotation) -
            yaw_radians(current_rotation)));
        const float alpha = yaw_error == 0.0F
            ? 1.0F
            : std::min(1.0F, kFlatTurnSpeedRadians * kDt / yaw_error);
        snapshot_.pose.rotations[root] = quat_nlerp_shortest(
            current_rotation, staging_root.rotation, alpha);
        snapshot_.pose.angular_velocities[root] = vec3(
            0.0F,
            shortest_angle(
                yaw_radians(snapshot_.pose.rotations[root]) -
                yaw_radians(current_rotation)) /
                kDt,
            0.0F);

        constexpr std::array<float, 3> kFutureSeconds = {
            0.32F, 0.68F, 1.00F};
        for (size_t sample = 0; sample < kFutureSeconds.size(); ++sample) {
            snapshot_.future_root_positions[sample] = step_planar_toward(
                next,
                staging_root.position,
                kFlatWalkSpeedMps * kFutureSeconds[sample]);
            snapshot_.future_root_rotations[sample] = staging_root.rotation;
        }
    }

private:
    interaction::LocomotionSnapshot snapshot_{};
};

interaction::RuntimeInput idle_input(
    const interaction::LocomotionSnapshot& locomotion) {
    interaction::RuntimeInput input{};
    input.dt = kDt;
    input.locomotion = locomotion;
    return input;
}

interaction::RuntimeInput pick_input(
    const interaction::LocomotionSnapshot& locomotion,
    interaction::PickRequest request) {
    interaction::RuntimeInput input = idle_input(locomotion);
    input.interact_pressed = true;
    input.pick_request = request;
    return input;
}

interaction::RuntimeInput place_input(
    const interaction::LocomotionSnapshot& locomotion,
    interaction::PlaceRequest request) {
    interaction::RuntimeInput input = idle_input(locomotion);
    input.interact_pressed = true;
    input.place_request = request;
    return input;
}

struct AttachmentAudit {
    bool attached = false;
    int attach_events = 0;
    int release_events = 0;

    void observe(const interaction::RuntimeOutput& output) {
        if (output.diagnostics.attached == attached) return;
        attached = output.diagnostics.attached;
        if (attached) {
            ++attach_events;
        } else {
            ++release_events;
            require(
                output.diagnostics.state ==
                    interaction::RuntimeState::PlaceRelease,
                "attachment released outside PlaceRelease");
        }
    }
};

class ProbeFixture {
public:
    ProbeFixture(
        interaction::Database database_value,
        const interaction::Features& features_value,
        interaction::RuntimeConfig config_value,
        const PositionMatrixRow& position_row_value =
            kPositionMatrixRows.front())
        : database(std::move(database_value)),
          features(features_value),
          config(config_value),
          position_row(position_row_value),
          live(initial_locomotion_for_fixture()) {
        interaction::InteractionTarget controller_authored =
            interaction::make_controller_demo_target(database);
        require_authored_slot_identity_mutations_are_observable(
            controller_authored);
        interaction::InteractionTarget authored = transform_pickup_scene(
            controller_authored, position_row);
        target = registry.upsert(std::move(authored));
        const interaction::InteractionTarget* registered = registry.find(target);
        require(registered != nullptr, "authored target registration failed");
        pick_affordance = registered->affordances.at(0).id;
        authored_target = *registered;

        const interaction::PlacementSurface controller_destination =
            interaction::make_controller_demo_destination_surface(
                database, controller_authored);
        const interaction::PlacementFit baseline_destination_fit =
            interaction::evaluate_placement_fit(
                controller_destination,
                controller_destination.affordances.at(0),
                controller_authored.object_bounds);
        interaction::PlacementSurface destination =
            transform_destination_scene(
                controller_destination, position_row);
        const interaction::PlacementFit fresh_destination_fit =
            interaction::evaluate_placement_fit(
                destination,
                destination.affordances.at(0),
                authored_target.object_bounds);
        require(
            baseline_destination_fit.accepted &&
                exact_placement_fit(
                    fresh_destination_fit, baseline_destination_fit),
            "rigid destination transform changed fresh placement fit");
        place_affordance = destination.affordances.at(0).id;
        destination_surface = surfaces.upsert(std::move(destination));
        const interaction::PlacementSurface* retained =
            surfaces.find(destination_surface);
        require(retained != nullptr, "destination surface registration failed");
        authored_destination = *retained;

        live = DeterministicFlatLiveProvider(
            initial_locomotion_for_row(
                database, authored_target, position_row));
        require_selected_initial_root(
            database, authored_target, position_row, live.snapshot());
        runtime = std::make_unique<interaction::InteractionRuntime>(
            database,
            features,
            registry,
            surfaces,
            place_library,
            config);
    }

    ProbeFixture(const ProbeFixture&) = delete;
    ProbeFixture& operator=(const ProbeFixture&) = delete;

    interaction::PlaceStagingPreview preview() {
        const interaction::PlaceStagingPreview result =
            runtime->preview_place(destination_surface, place_affordance);
        if (result.accepted) {
            preview_ids.push_back(result.candidate.selection_id);
        }
        return result;
    }

    bool owns_preview_id(uint64_t id) const {
        return id != 0U &&
            std::find(preview_ids.begin(), preview_ids.end(), id) !=
                preview_ids.end();
    }

    interaction::PickRequest pickup_request(uint64_t request_id) const {
        return {target, pick_affordance, request_id};
    }

    interaction::PlaceRequest placement_request(
        uint64_t request_id,
        uint64_t selection_id) const {
        return {
            target,
            destination_surface,
            place_affordance,
            request_id,
            selection_id,
        };
    }

    interaction::Database database;
    interaction::Features features;
    interaction::RuntimeConfig config;
    interaction::TargetRegistry registry;
    interaction::PlacementSurfaceRegistry surfaces;
    interaction::PlaceMotionLibrary place_library{};
    interaction::TargetHandle target{};
    uint32_t pick_affordance = 0U;
    interaction::SurfaceHandle destination_surface{};
    uint32_t place_affordance = 0U;
    interaction::InteractionTarget authored_target{};
    interaction::PlacementSurface authored_destination{};
    PositionMatrixRow position_row{};
    DeterministicFlatLiveProvider live;
    std::unique_ptr<interaction::InteractionRuntime> runtime;
    std::vector<uint64_t> preview_ids;
    int actual_carry_staging_runs = 0;
    int preflight_recovery_runs = 0;

private:
    interaction::LocomotionSnapshot initial_locomotion_for_fixture() const {
        const interaction::InteractionTarget authored =
            interaction::make_controller_demo_target(database);
        const interaction::InteractionTarget target_for_mapping =
            transform_pickup_scene(authored, position_row);
        return initial_locomotion_for_row(
            database, target_for_mapping, position_row);
    }
};

interaction::RuntimeOutput run_pickup_to_carry(
    ProbeFixture& fixture,
    AttachmentAudit& audit,
    uint64_t request_id) {
    using interaction::Reason;
    using interaction::ResultCode;
    using interaction::RuntimeState;

    interaction::RuntimeOutput output = fixture.runtime->update(pick_input(
        fixture.live.snapshot(), fixture.pickup_request(request_id)));
    audit.observe(output);
    require(
        output.diagnostics.state == RuntimeState::Preflight,
        "pickup did not publish Preflight");
    for (int update = 0;
         update < kMaximumPickupUpdates &&
         output.diagnostics.state != RuntimeState::Carry;
         ++update) {
        output = fixture.runtime->update(idle_input(fixture.live.snapshot()));
        audit.observe(output);
        require(
            output.diagnostics.result != ResultCode::Failed &&
                output.diagnostics.result != ResultCode::Rejected,
            "pickup failed before Carry");
        require(
            output.diagnostics.reason == Reason::None,
            "pickup reported a non-None reason");
    }
    require(
        output.diagnostics.state == RuntimeState::Carry,
        "pickup timed out before Carry");
    require(output.diagnostics.attached, "Carry is not attached");
    require(
        output.diagnostics.object_state == interaction::ObjectState::Held,
        "Carry object is not Held");
    require(
        output.diagnostics.clip == 0,
        "deterministic pickup did not select authored clip 0");
    require(
        audit.attach_events == 1 && audit.release_events == 0,
        "pickup attachment transition count is not exactly one");
    const interaction::InteractionTarget* held = fixture.registry.find(
        fixture.target);
    require(
        held != nullptr && held->state == interaction::ObjectState::Held &&
            held->owner_request == request_id,
        "registry did not retain the pickup owner in Carry");
    return output;
}

interaction::Reason run_height_rejection_control(
    const interaction::Database& database,
    const interaction::Features& features,
    interaction::RuntimeConfig config,
    const PositionMatrixRow& successful_row,
    const char* label,
    float destination_height_m) {
    PositionMatrixRow control = successful_row;
    control.label = label;
    control.destination_translation.y = destination_height_m;
    auto fixture = std::make_unique<ProbeFixture>(
        database, features, config, control);
    AttachmentAudit audit{};
    const interaction::RuntimeOutput carry = run_pickup_to_carry(
        *fixture, audit, 8101U);
    const interaction::PlaceStagingPreview preview = fixture->preview();
    require(
        carry.diagnostics.state == interaction::RuntimeState::Carry &&
            carry.diagnostics.attached &&
            audit.attach_events == 1 && audit.release_events == 0 &&
            !preview.accepted && !preview.ready &&
            preview.reason == interaction::Reason::PlacementOutOfBounds,
        std::string(label) +
            " did not preserve the signed-height capability limit");
    return preview.reason;
}

void require_preview_configuration(
    const interaction::PlaceStagingPreview& preview,
    const interaction::RuntimeConfig& config) {
    if (!preview.accepted) {
        fail(
            "runtime placement preview was rejected: " +
            std::string(reason_name(preview.reason)));
    }
    require(
        preview.candidate.mode ==
            interaction::PlaceMotionMode::ReversedPickup,
        "runtime placement preview selected an unexpected mode");
    require(
        preview.candidate.selection_id != 0U,
        "runtime placement preview has a zero selection ID");
    require(
        preview.ik_config_fingerprint != 0U,
        "runtime placement preview has a zero IK fingerprint");
    require(
        exact(preview.ik, config.ik) &&
            exact(preview.candidate.ik, config.ik),
        "runtime placement preview did not bind the complete IK config");
}

void require_expected_reverse_source(
    const ProbeFixture& fixture,
    const interaction::RuntimeOutput& carry,
    const interaction::PlaceStagingPreview& preview,
    const CertifiedFixtureFrames& frames) {
    require_preview_configuration(preview, fixture.config);
    require(
        preview.candidate.clip == carry.diagnostics.clip,
        "place preview did not retain the pickup clip");
    require(
        preview.candidate.entry_frame == frames.reverse_start,
        "place preview did not choose the earliest certified Hold window");
    require(
        preview.candidate.release_frame == frames.contact,
        "place preview did not derive release from Contact");
    require(
        preview.candidate.direction == -1,
        "reversed pickup did not run backward");

    const interaction::GraspAffordance* grasp =
        fixture.registry.find_affordance(
            fixture.target, fixture.pick_affordance);
    const interaction::PlaceAffordance* place =
        fixture.surfaces.find_affordance(
            fixture.destination_surface, fixture.place_affordance);
    const interaction::PlacementSurface* surface =
        fixture.surfaces.find(fixture.destination_surface);
    require(
        grasp != nullptr && place != nullptr && surface != nullptr,
        "retained destination or held affordance became unavailable");
    const interaction::Transform goal_object =
        interaction::placement_goal_world(
            *surface, place->object_in_surface);
    const interaction::Transform goal_hand = interaction::compose(
        goal_object, grasp->hand_in_object);
    const interaction::Transform source_hand = hand_transform(
        interaction::pose_at_frame(fixture.database, frames.contact),
        grasp->hand);
    const interaction::Transform mapped_goal_hand = interaction::compose(
        preview.candidate.scene_from_source, source_hand);
    require(
        near(mapped_goal_hand, goal_hand),
        "runtime preview scene mapping missed the destination hand goal");
    const interaction::Transform expected_staging = interaction::compose(
        preview.candidate.scene_from_source,
        root_transform(interaction::pose_at_frame(
            fixture.database, frames.reverse_start)));
    require(
        near(preview.staging_root_world, expected_staging) &&
            near(preview.candidate.staging_root_world, expected_staging),
        "runtime preview staging root differs from certified source root");
}

struct StagedCarry {
    interaction::RuntimeOutput output{};
    interaction::PlaceStagingPreview far{};
    interaction::PlaceStagingPreview staged{};
    vec3 carry_start_root{};
    interaction::Transform carry_start_object{};
    float root_displacement_m = 0.0F;
    float object_displacement_m = 0.0F;
};

StagedCarry run_actual_carry_staging(
    ProbeFixture& fixture,
    const CertifiedFixtureFrames& frames,
    AttachmentAudit& audit,
    uint64_t pickup_request_id) {
    using interaction::RuntimeState;
    require(
        fixture.actual_carry_staging_runs == 0,
        "fixture repeated actual Carry staging");
    ++fixture.actual_carry_staging_runs;
    StagedCarry staged{};
    staged.output = run_pickup_to_carry(
        fixture, audit, pickup_request_id);
    staged.carry_start_root = staged.output.pose.positions[
        g1_skeleton::Simulation];
    staged.carry_start_object = staged.output.object_world;
    staged.far = fixture.preview();
    require_expected_reverse_source(
        fixture, staged.output, staged.far, frames);
    require(
        !staged.far.ready &&
            staged.far.root_error_m >
                fixture.config.place.match.maximum_entry_root_error_m,
        "initial runtime preview is not accepted-far/not-ready");
    require(
        fixture.owns_preview_id(staged.far.candidate.selection_id),
        "far selection ID did not originate from runtime preview");

    for (int update = 0; update < kMaximumStagingUpdates; ++update) {
        fixture.live.advance_toward(
            staged.output, staged.far.staging_root_world);
        staged.output = fixture.runtime->update(
            idle_input(fixture.live.snapshot()));
        audit.observe(staged.output);
        require(
            staged.output.diagnostics.state == RuntimeState::Carry &&
                staged.output.diagnostics.attached,
            "actual runtime left attached Carry during staging");
        staged.staged = fixture.preview();
        require_preview_configuration(staged.staged, fixture.config);
        require(
            near(
                staged.staged.staging_root_world,
                staged.far.staging_root_world),
            "runtime staging root changed while Carry moved");
        require(
            staged.staged.ik_config_fingerprint ==
                staged.far.ik_config_fingerprint &&
                exact(staged.staged.ik, staged.far.ik),
            "movement changed runtime IK configuration identity");
        if (staged.staged.ready) break;
    }
    require(staged.staged.ready, "actual Carry staging timed out");
    require(
        fixture.owns_preview_id(staged.staged.candidate.selection_id),
        "ready selection ID did not originate from runtime preview");
    require(
        staged.staged.candidate.selection_id !=
            staged.far.candidate.selection_id,
        "movement did not change runtime selection identity");
    staged.root_displacement_m = planar_distance(
        staged.carry_start_root,
        staged.output.pose.positions[g1_skeleton::Simulation]);
    staged.object_displacement_m = planar_distance(
        staged.carry_start_object.position,
        staged.output.object_world.position);
    require(
        staged.root_displacement_m > 0.20F &&
            staged.object_displacement_m > 0.20F,
        "staged pose/object did not come from displaced actual Carry");
    require(
        audit.attach_events == 1 && audit.release_events == 0,
        "Carry staging changed attachment ownership");
    return staged;
}

interaction::RuntimeOutput require_two_frame_preflight_recovery(
    ProbeFixture& fixture,
    const interaction::RuntimeOutput& staged,
    uint64_t selection_id,
    uint64_t request_id) {
    using interaction::Reason;
    using interaction::ResultCode;
    using interaction::RuntimeState;
    require(
        fixture.actual_carry_staging_runs == 1 &&
            fixture.preflight_recovery_runs == 0,
        "negative recovery receiver is not independently staged and clean");
    ++fixture.preflight_recovery_runs;

    const interaction::InteractionTarget before =
        *fixture.registry.find(fixture.target);
    interaction::RuntimeOutput edge = fixture.runtime->update(place_input(
        fixture.live.snapshot(),
        fixture.placement_request(request_id, selection_id)));
    require(
        edge.diagnostics.state == RuntimeState::PlacePreflight &&
            edge.diagnostics.attached,
        "invalid runtime selection did not publish PlacePreflight");
    require(
        exact(edge.pose, staged.pose) &&
            exact(edge.object_world, staged.object_world),
        "invalid selection changed frozen Carry on preflight edge");

    interaction::RuntimeOutput recovery = fixture.runtime->update(
        idle_input(fixture.live.snapshot()));
    require(
        recovery.diagnostics.state == RuntimeState::Carry &&
            recovery.diagnostics.result == ResultCode::Rejected &&
            recovery.diagnostics.reason == Reason::TargetChanged &&
            recovery.diagnostics.attached,
        "invalid runtime selection did not recover in two frames");
    require(
        exact(recovery.pose, edge.pose) &&
            exact(recovery.object_world, edge.object_world),
        "invalid selection changed Carry during recovery");
    const interaction::InteractionTarget* after = fixture.registry.find(
        fixture.target);
    require(
        after != nullptr && after->handle == before.handle &&
            after->state == before.state &&
            after->owner_request == before.owner_request &&
            exact(after->object_world, before.object_world),
        "invalid selection changed held registry state");
    return recovery;
}

std::string_view state_name(interaction::RuntimeState state) {
    using interaction::RuntimeState;
    switch (state) {
    case RuntimeState::Carry: return "Carry";
    case RuntimeState::PlacePreflight: return "PlacePreflight";
    case RuntimeState::PlaceAlign: return "PlaceAlign";
    case RuntimeState::PlaceReplay: return "PlaceReplay";
    case RuntimeState::PlaceRelease: return "PlaceRelease";
    case RuntimeState::Locomotion: return "Locomotion";
    default: fail("unexpected state in successful place sequence");
    }
}

void append_state(
    std::vector<interaction::RuntimeState>& states,
    interaction::RuntimeState state) {
    if (states.empty() || states.back() != state) states.push_back(state);
}

struct PlacementEvidence {
    interaction::RuntimeOutput final_output{};
    std::vector<interaction::RuntimeState> states;
    uint64_t ik_fingerprint = 0U;
    int32_t release_frame = -1;
    int32_t reverse_start_frame = -1;
    bool actual_fit = false;
    bool repick_support_is_destination = false;
};

PlacementEvidence run_successful_place_and_repick(
    ProbeFixture& fixture,
    const StagedCarry& staged,
    AttachmentAudit& audit,
    bool verify_repick_preflight = true) {
    using interaction::ObjectState;
    using interaction::Reason;
    using interaction::ResultCode;
    using interaction::RuntimeState;

    require(
        fixture.owns_preview_id(staged.staged.candidate.selection_id),
        "successful place ID is not runtime-owned");
    PlacementEvidence evidence{};
    evidence.ik_fingerprint = staged.staged.ik_config_fingerprint;
    evidence.release_frame = staged.staged.candidate.release_frame;
    evidence.reverse_start_frame = staged.staged.candidate.entry_frame;
    evidence.states.push_back(RuntimeState::Carry);

    interaction::RuntimeOutput output = fixture.runtime->update(place_input(
        fixture.live.snapshot(),
        fixture.placement_request(
            8201U, staged.staged.candidate.selection_id)));
    audit.observe(output);
    append_state(evidence.states, output.diagnostics.state);
    require(
        output.diagnostics.state == RuntimeState::PlacePreflight &&
            exact(output.pose, staged.output.pose) &&
            exact(output.object_world, staged.output.object_world),
        "current ready runtime ID did not freeze PlacePreflight");

    bool saw_release = false;
    interaction::Transform release_object{};
    interaction::TargetHandle placed_handle{};
    for (int update = 0; update < kMaximumPlaceUpdates; ++update) {
        output = fixture.runtime->update(idle_input(fixture.live.snapshot()));
        audit.observe(output);
        append_state(evidence.states, output.diagnostics.state);
        require(
            output.diagnostics.state == RuntimeState::PlaceAlign ||
                output.diagnostics.state == RuntimeState::PlaceReplay ||
                output.diagnostics.state == RuntimeState::PlaceRelease ||
                output.diagnostics.state == RuntimeState::Locomotion,
            "runtime published an unexpected successful place state");
        require(
            output.diagnostics.result != ResultCode::Failed &&
                output.diagnostics.result != ResultCode::Rejected &&
                output.diagnostics.reason == Reason::None,
            "successful place reported failure/rejection");
        if (output.diagnostics.state != RuntimeState::Locomotion) {
            require(
                output.diagnostics.place.mode ==
                    interaction::PlaceMotionMode::ReversedPickup,
                "successful place mode changed after preflight");
            require(
                output.diagnostics.place.ik_config_fingerprint ==
                    staged.staged.ik_config_fingerprint &&
                    exact(
                        output.diagnostics.place.effective_ik,
                        fixture.config.ik),
                "successful place lost runtime IK identity");
        }
        if (output.diagnostics.state == RuntimeState::PlaceAlign) {
            require(
                output.diagnostics.place.preflight_config_identity,
                "Place preflight did not bind exact runtime config");
        }
        if (output.diagnostics.state == RuntimeState::PlaceRelease &&
            !saw_release) {
            saw_release = true;
            release_object = output.object_world;
            placed_handle = output.diagnostics.target;
            require(
                output.diagnostics.place.source_frame ==
                    evidence.release_frame,
                "runtime released on an unexpected source frame");
            require(
                output.diagnostics.place.actual_fit.accepted,
                "runtime release did not publish accepted actual fit");
            require(
                !output.diagnostics.attached &&
                    output.diagnostics.object_state == ObjectState::Free,
                "PlaceRelease did not atomically detach the object");
            const interaction::InteractionTarget* released =
                fixture.registry.find(placed_handle);
            require(
                released != nullptr &&
                    exact(released->object_world, release_object),
                "registry release pose differs from runtime release pose");
        }
        if (output.diagnostics.state == RuntimeState::Locomotion) break;
    }
    require(saw_release, "place never reached PlaceRelease");
    require(
        output.diagnostics.state == RuntimeState::Locomotion,
        "place timed out before Locomotion");
    const std::vector<RuntimeState> expected_states = {
        RuntimeState::Carry,
        RuntimeState::PlacePreflight,
        RuntimeState::PlaceAlign,
        RuntimeState::PlaceReplay,
        RuntimeState::PlaceRelease,
        RuntimeState::Locomotion,
    };
    require(
        evidence.states == expected_states,
        "successful place state sequence is not exact");
    require(
        audit.attach_events == 1 && audit.release_events == 1 &&
            !audit.attached,
        "successful place attachment transitions are not exact");
    require(
        output.diagnostics.result == ResultCode::Succeeded &&
            output.diagnostics.reason == Reason::None &&
            !output.diagnostics.attached &&
            output.diagnostics.object_state == ObjectState::Free,
        "final place result is not detached success");
    require(
        output.diagnostics.target == placed_handle &&
            placed_handle.id == fixture.target.id &&
            placed_handle.generation == fixture.target.generation + 1U,
        "place did not publish the returned target generation");

    const interaction::InteractionTarget* placed = fixture.registry.find(
        placed_handle);
    const interaction::PlacementSurface* destination = fixture.surfaces.find(
        fixture.destination_surface);
    const interaction::PlaceAffordance* place_affordance =
        fixture.surfaces.find_affordance(
            fixture.destination_surface, fixture.place_affordance);
    require(
        placed != nullptr && destination != nullptr &&
            place_affordance != nullptr,
        "released target or retained destination became unavailable");
    const interaction::Transform transformed_goal =
        interaction::placement_goal_world(
            *destination, place_affordance->object_in_surface);
    require(
        exact(placed->object_world, release_object),
        "released target differs from the runtime release pose");
    require(
        exact(placed->table_world, destination->support_volume_world) &&
            exact(placed->table_size, destination->support_volume_size) &&
            exact(
                destination->surface_world,
                fixture.authored_destination.surface_world) &&
            exact(
                destination->support_volume_world,
                fixture.authored_destination.support_volume_world),
        "released target did not commit transformed destination context");
    require(
        near(placed->object_world, transformed_goal),
        "released target did not finish at the transformed goal");
    const interaction::PlacementFit fit =
        interaction::evaluate_actual_placement_fit(
            *destination,
            *place_affordance,
            placed->object_world,
            placed->object_bounds);
    require(fit.accepted, "released target failed fresh actual-fit check");
    evidence.actual_fit = true;
    if (!verify_repick_preflight) {
        evidence.final_output = output;
        return evidence;
    }

    const interaction::GraspAffordance* repick_affordance =
        fixture.registry.find_affordance(
            placed_handle, fixture.pick_affordance);
    require(
        repick_affordance != nullptr,
        "released generation lost pickup affordance");

    const int32_t repick_contact = first_phase_frame(
        fixture.database, 0U, interaction::Phase::Contact);
    const int32_t repick_entry = first_phase_frame(
        fixture.database, 0U, interaction::Phase::Reach);
    const interaction::Transform source_rest_object = frame_transform(
        fixture.database.object_positions,
        fixture.database.object_rotations,
        static_cast<size_t>(repick_contact - 1));
    const interaction::Transform repick_scene = planar_alignment(
        source_rest_object, placed->object_world);
    const interaction::Transform mapped_entry_root = interaction::compose(
        repick_scene,
        root_transform(interaction::pose_at_frame(
            fixture.database, repick_entry)));
    interaction::RuntimeOutput free_locomotion = output;
    bool repick_locomotion_settled = false;
    for (int update = 0; update < 64; ++update) {
        fixture.live.advance_toward(free_locomotion, mapped_entry_root);
        free_locomotion = fixture.runtime->update(
            idle_input(fixture.live.snapshot()));
        const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
        const interaction::Transform settled_root = root_transform(
            fixture.live.snapshot().pose);
        require(
            free_locomotion.diagnostics.state == RuntimeState::Locomotion &&
                free_locomotion.diagnostics.object_state == ObjectState::Free &&
                exact(free_locomotion.object_world, placed->object_world),
            "native-rate post-release Locomotion changed placed state");
        if (planar_distance(
                settled_root.position, mapped_entry_root.position) == 0.0F &&
            shortest_angle(
                yaw_radians(settled_root.rotation) -
                yaw_radians(mapped_entry_root.rotation)) == 0.0F &&
            length(fixture.live.snapshot().pose.velocities[root]) == 0.0F &&
            length(
                fixture.live.snapshot().pose.angular_velocities[root]) ==
                0.0F) {
            repick_locomotion_settled = true;
            break;
        }
    }
    require(
        repick_locomotion_settled,
        std::string(fixture.position_row.label) +
            " native-rate post-release Locomotion did not settle for repick");

    interaction::QueryInput repick_query{};
    repick_query.locomotion = fixture.live.snapshot();
    repick_query.grasp_world = interaction::compose(
        placed->object_world, repick_affordance->hand_in_object);
    repick_query.table_world = placed->table_world;
    repick_query.table_size = placed->table_size;
    repick_query.approach_direction_object =
        repick_affordance->approach_direction_object;
    repick_query.object_dimensions = placed->object_dimensions;
    repick_query.hand = repick_affordance->hand;
    require(
        exact(repick_query.table_world, destination->support_volume_world) &&
            exact(repick_query.table_size, destination->support_volume_size) &&
            !exact(repick_query.table_world, fixture.authored_target.table_world),
        "fresh pick query did not use destination support context");
    static_cast<void>(interaction::build_raw_query(repick_query));
    evidence.repick_support_is_destination = true;

    const interaction::PickRequest repick{
        placed_handle, fixture.pick_affordance, 8202U};
    interaction::RuntimeOutput repick_output = fixture.runtime->update(
        pick_input(fixture.live.snapshot(), repick));
    require(
        repick_output.diagnostics.state == RuntimeState::Preflight,
        "released generation did not enter fresh pick Preflight");
    repick_output = fixture.runtime->update(
        idle_input(fixture.live.snapshot()));
    require(
        repick_output.diagnostics.state == RuntimeState::Align &&
            repick_output.diagnostics.target == placed_handle,
        "released generation did not pass fresh destination pick preflight");
    evidence.final_output = output;
    return evidence;
}

void print_json(
    const PlacementEvidence& evidence,
    int attachment_transitions,
    interaction::Reason positive_height_result,
    interaction::Reason negative_height_result) {
    std::cout
        << "{\"actual_carry_staging\":true"
        << ",\"actual_fit\":" << (evidence.actual_fit ? "true" : "false")
        << ",\"attachment_transitions\":" << attachment_transitions
        << ",\"destination_selected_directly\":true"
        << ",\"far_preview_accepted\":true"
        << ",\"far_preview_ready\":false"
        << ",\"final_attached\":"
        << (evidence.final_output.diagnostics.attached ? "true" : "false")
        << ",\"final_object_state\":\"Free\""
        << ",\"final_reason\":\"None\""
        << ",\"final_result\":\"Succeeded\""
        << ",\"ik_config_bound\":true"
        << ",\"ik_config_fingerprint\":" << evidence.ik_fingerprint
        << ",\"mode\":\"reversed_pickup\""
        << ",\"position_height_controls\":2"
        << ",\"position_height_negative_result\":\""
        << reason_name(negative_height_result) << '"'
        << ",\"position_height_positive_result\":\""
        << reason_name(positive_height_result) << '"'
        << ",\"position_matrix_cases\":" << kPositionMatrixRows.size()
        << ",\"release_frame\":" << evidence.release_frame
        << ",\"repick_support_is_destination\":"
        << (evidence.repick_support_is_destination ? "true" : "false")
        << ",\"reverse_start_frame\":" << evidence.reverse_start_frame
        << ",\"runtime_preview_only\":true"
        << ",\"staged_preview_ready\":true"
        << ",\"staged_selection_id_changed\":true"
        << ",\"state_sequence\":[";
    for (size_t index = 0; index < evidence.states.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << '\"' << state_name(evidence.states[index]) << '\"';
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string_view(argv[2]) != "--json") {
        std::cerr << "usage: interaction_place_probe <pack> --json\n";
        return 2;
    }

    try {
        const std::filesystem::path pack = argv[1];
        interaction::Database database = interaction::load_database(
            pack / "interaction_database.bin");
        const interaction::Features features = interaction::load_features(
            pack / "interaction_features.bin");
        validate_native_pack(database, features);

        const CertifiedFixtureFrames frames =
            make_first_hold_window_unstable(database);
        interaction::Database last_contact_mutation = database;
        const interaction::Hand active_hand =
            database.active_hands.at(0U) == 0U
            ? interaction::Hand::Left
            : interaction::Hand::Right;
        const size_t contact_index =
            static_cast<size_t>(frames.last_contact) * 2U +
            static_cast<size_t>(active_hand);
        require(
            last_contact_mutation.hand_contacts.at(contact_index) == 1U,
            "late-contact mutation source was not contact");
        last_contact_mutation.hand_contacts.at(contact_index) = 0U;
        size_t contact_differences = 0U;
        for (size_t index = 0;
             index < database.hand_contacts.size();
             ++index) {
            if (database.hand_contacts[index] !=
                last_contact_mutation.hand_contacts[index]) {
                ++contact_differences;
            }
        }
        require(
            contact_differences == 1U,
            "late-contact fixture changed more than one contact sample");

        const interaction::RuntimeConfig config{};
        auto success = std::make_unique<ProbeFixture>(
            database, features, config);
        AttachmentAudit success_audit{};
        const StagedCarry success_staged = run_actual_carry_staging(
            *success, frames, success_audit, 8101U);

        auto mutation = std::make_unique<ProbeFixture>(
            last_contact_mutation, features, config);
        AttachmentAudit mutation_audit{};
        const StagedCarry mutation_staged = run_actual_carry_staging(
            *mutation, frames, mutation_audit, 8101U);
        require(
            mutation_staged.far.candidate.entry_frame ==
                success_staged.far.candidate.entry_frame &&
                mutation_staged.far.candidate.selection_id ==
                    success_staged.far.candidate.selection_id,
            "late contact changed certified reverse selection");

        auto stale_receiver = std::make_unique<ProbeFixture>(
            database, features, config);
        AttachmentAudit stale_receiver_audit{};
        StagedCarry stale_receiver_staged = run_actual_carry_staging(
            *stale_receiver, frames, stale_receiver_audit, 8101U);
        require(
            stale_receiver->owns_preview_id(
                stale_receiver_staged.far.candidate.selection_id),
            "stale far ID lacks runtime-preview provenance");
        stale_receiver_staged.output = require_two_frame_preflight_recovery(
            *stale_receiver,
            stale_receiver_staged.output,
            stale_receiver_staged.far.candidate.selection_id,
            8301U);
        const interaction::PlaceStagingPreview stale_recovered_ready =
            stale_receiver->preview();
        require(
            stale_recovered_ready.accepted && stale_recovered_ready.ready,
            "stale-ID recovery did not preserve ready Carry preview");

        auto mismatch_receiver = std::make_unique<ProbeFixture>(
            database, features, config);
        AttachmentAudit mismatch_receiver_audit{};
        StagedCarry mismatch_receiver_staged = run_actual_carry_staging(
            *mismatch_receiver, frames, mismatch_receiver_audit, 8101U);

        interaction::RuntimeConfig mismatched_config = config;
        mismatched_config.ik.damping = std::nextafter(
            mismatched_config.ik.damping,
            std::numeric_limits<float>::infinity());
        auto mismatch_producer = std::make_unique<ProbeFixture>(
            database, features, mismatched_config);
        AttachmentAudit mismatch_producer_audit{};
        const StagedCarry mismatch_producer_staged =
            run_actual_carry_staging(
            *mismatch_producer, frames, mismatch_producer_audit, 8101U);
        require(
            mismatch_producer->owns_preview_id(
                mismatch_producer_staged.staged.candidate.selection_id) &&
                mismatch_producer_staged.staged.ik_config_fingerprint !=
                    mismatch_receiver_staged.staged.ik_config_fingerprint &&
                mismatch_producer_staged.staged.candidate.selection_id !=
                    mismatch_receiver_staged.staged.candidate.selection_id,
            "IK-mismatched ID lacks distinct runtime-preview provenance");
        mismatch_receiver_staged.output = require_two_frame_preflight_recovery(
            *mismatch_receiver,
            mismatch_receiver_staged.output,
            mismatch_producer_staged.staged.candidate.selection_id,
            8302U);
        require(
            mismatch_receiver_staged.output.diagnostics.state ==
                interaction::RuntimeState::Carry,
            "IK-mismatch recovery did not return to Carry");

        const PlacementEvidence evidence = run_successful_place_and_repick(
            *success, success_staged, success_audit);
        require(
            evidence.release_frame == frames.contact &&
                evidence.reverse_start_frame == frames.reverse_start,
            "final evidence frames differ from fixture-derived frames");
        for (size_t index = 1U;
             index < kPositionMatrixRows.size();
             ++index) {
            const PositionMatrixRow& row = kPositionMatrixRows[index];
            auto matrix_case = std::make_unique<ProbeFixture>(
                database, features, config, row);
            AttachmentAudit matrix_audit{};
            const StagedCarry matrix_staged = run_actual_carry_staging(
                *matrix_case, frames, matrix_audit, 8101U);
            const PlacementEvidence matrix_evidence =
                run_successful_place_and_repick(
                    *matrix_case, matrix_staged, matrix_audit, false);
            require(
                matrix_evidence.states == evidence.states &&
                    matrix_evidence.release_frame == evidence.release_frame &&
                    matrix_evidence.reverse_start_frame ==
                        evidence.reverse_start_frame &&
                    matrix_evidence.ik_fingerprint ==
                        evidence.ik_fingerprint &&
                    matrix_evidence.actual_fit &&
                    matrix_audit.attach_events == 1 &&
                    matrix_audit.release_events == 1 &&
                    !matrix_audit.attached,
                std::string(row.label) +
                    " did not match baseline placement evidence");
        }
        const interaction::Reason positive_height_result =
            run_height_rejection_control(
                database,
                features,
                config,
                kPositionMatrixRows[1],
                "positive_height_control",
                0.05F);
        const interaction::Reason negative_height_result =
            run_height_rejection_control(
                database,
                features,
                config,
                kPositionMatrixRows[2],
                "negative_height_control",
                -0.05F);
        print_json(
            evidence,
            success_audit.attach_events,
            positive_height_result,
            negative_height_result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interaction_place_probe: " << error.what() << '\n';
        return 1;
    }
}
