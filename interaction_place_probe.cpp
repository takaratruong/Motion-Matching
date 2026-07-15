#include "interaction_controller_adapter.h"

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

interaction::InteractionTarget make_headless_target(
    const interaction::Database& database) {
    interaction::InteractionTarget target =
        interaction::make_controller_demo_target(database);
    require(
        target.affordances.size() == 1U,
        "headless fixture requires one pickup affordance");
    const int32_t contact = first_phase_frame(
        database, 0U, interaction::Phase::Contact);
    const interaction::Transform source_contact_hand = hand_transform(
        interaction::pose_at_frame(database, contact),
        target.affordances.front().hand);
    const interaction::Transform source_rest_object = frame_transform(
        database.object_positions,
        database.object_rotations,
        static_cast<size_t>(contact - 1));
    target.affordances.front().hand_in_object = interaction::compose(
        interaction::inverse(source_rest_object), source_contact_hand);
    return target;
}

interaction::PlacementSurface make_headless_destination_surface(
    const interaction::Database& database,
    const interaction::InteractionTarget& source_target) {
    interaction::PlacementSurface destination =
        interaction::make_controller_demo_destination_surface(
            database, source_target);
    require(
        destination.affordances.size() == 1U,
        "headless fixture requires one placement affordance");
    const int32_t contact = first_phase_frame(
        database, 0U, interaction::Phase::Contact);
    const interaction::Transform source_table = frame_transform(
        database.table_positions, database.table_rotations, 0U);
    const vec3 source_table_size = read_vec3(database.table_sizes, 0U);
    const interaction::Transform source_surface = interaction::compose(
        source_table,
        interaction::Transform{
            vec3(0.0F, 0.5F * source_table_size.y, 0.0F), quat()});
    const interaction::Transform source_rest_object = frame_transform(
        database.object_positions,
        database.object_rotations,
        static_cast<size_t>(contact - 1));
    const vec3 source_normal = quat_mul_vec3(
        source_table.rotation, vec3(0.0F, 1.0F, 0.0F));
    const float projection_distance = dot(
        source_surface.position - source_rest_object.position,
        source_normal);
    const vec3 projected_support = source_rest_object.position +
        projection_distance * source_normal;

    interaction::PlaceAffordance& place = destination.affordances.front();
    place.support_point_object = interaction::compose(
        interaction::inverse(source_rest_object),
        interaction::Transform{projected_support, quat()}).position;
    const interaction::Transform rest_object_in_surface =
        interaction::compose(
            interaction::inverse(source_surface), source_rest_object);
    place.object_in_surface = rest_object_in_surface;
    interaction::PlacementFit rest_fit =
        interaction::evaluate_placement_fit(
            destination, place, source_target.object_bounds);
    if (!rest_fit.accepted) {
        require(
            rest_fit.footprint_valid && rest_fit.overhead_valid &&
                rest_fit.support_gap_m == 0.0F &&
                rest_fit.lowest_corner_m < 0.0F,
            "rest-authored destination has an unexpected fit failure");
        const float bounds_clearance = -rest_fit.lowest_corner_m;
        const vec3 destination_normal = quat_mul_vec3(
            destination.surface_world.rotation,
            vec3(0.0F, 1.0F, 0.0F));
        destination.surface_world.position =
            destination.surface_world.position -
            bounds_clearance * destination_normal;
        destination.support_volume_world.position =
            destination.support_volume_world.position -
            bounds_clearance * destination_normal;
        place.object_in_surface.position.y += bounds_clearance;
        rest_fit = interaction::evaluate_placement_fit(
            destination, place, source_target.object_bounds);
    }
    require(
        rest_fit.accepted,
        "rest-authored destination slot is not support-fit");
    return destination;
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
        interaction::RuntimeConfig config_value)
        : database(std::move(database_value)),
          features(features_value),
          config(config_value),
          live(initial_locomotion_for_fixture()) {
        interaction::InteractionTarget authored =
            make_headless_target(database);
        target = registry.upsert(std::move(authored));
        const interaction::InteractionTarget* registered = registry.find(target);
        require(registered != nullptr, "authored target registration failed");
        pick_affordance = registered->affordances.at(0).id;
        authored_target = *registered;

        interaction::PlacementSurface destination =
            make_headless_destination_surface(
                database, authored_target);
        place_affordance = destination.affordances.at(0).id;
        destination_surface = surfaces.upsert(std::move(destination));
        const interaction::PlacementSurface* retained =
            surfaces.find(destination_surface);
        require(retained != nullptr, "destination surface registration failed");
        authored_destination = *retained;

        live = DeterministicFlatLiveProvider(
            initial_locomotion(database, authored_target));
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
    DeterministicFlatLiveProvider live;
    std::unique_ptr<interaction::InteractionRuntime> runtime;
    std::vector<uint64_t> preview_ids;
    int actual_carry_staging_runs = 0;
    int preflight_recovery_runs = 0;

private:
    interaction::LocomotionSnapshot initial_locomotion_for_fixture() const {
        interaction::InteractionTarget target_for_mapping =
            make_headless_target(database);
        return initial_locomotion(database, target_for_mapping);
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
    const interaction::Transform expected_scene = planar_alignment(
        source_hand, goal_hand);
    require(
        near(preview.candidate.scene_from_source, expected_scene),
        "runtime preview scene mapping differs from source/destination");
    const interaction::Transform expected_staging = interaction::compose(
        expected_scene,
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
    AttachmentAudit& audit) {
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
    require(
        exact(placed->object_world, release_object) &&
            exact(placed->table_world, destination->support_volume_world) &&
            exact(placed->table_size, destination->support_volume_size),
        "released target did not commit destination support context");
    const interaction::PlacementFit fit =
        interaction::evaluate_actual_placement_fit(
            *destination,
            *place_affordance,
            placed->object_world,
            placed->object_bounds);
    require(fit.accepted, "released target failed fresh actual-fit check");
    evidence.actual_fit = true;

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
        "native-rate post-release Locomotion did not settle for repick");

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
    evidence.repick_support_is_destination = true;
    evidence.final_output = output;
    return evidence;
}

void print_json(
    const PlacementEvidence& evidence,
    int attachment_transitions) {
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
        print_json(evidence, success_audit.attach_events);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interaction_place_probe: " << error.what() << '\n';
        return 1;
    }
}
