#include "interaction_runtime.h"
#include "tests/cpp/pick_entry_oracle_roots.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

constexpr float kExpectedReachCost = 0.666622F;
constexpr float kExpectedPlusCost = 0.718336F;
constexpr float kCostTolerance = 2.0e-6F;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
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
    size_t index) {
    return {read_vec3(positions, index), read_quat(rotations, index)};
}

int32_t first_phase_frame(
    const interaction::Database& database,
    interaction::Phase phase) {
    require(
        database.clip_count != 0U &&
        !database.range_starts.empty() &&
        !database.range_stops.empty(),
        "oracle pack has no clip 0");
    const int32_t start = database.range_starts.at(0U);
    const int32_t stop = database.range_stops.at(0U);
    for (int32_t frame = start; frame < stop; ++frame) {
        if (database.phases.at(static_cast<size_t>(frame)) ==
            static_cast<uint8_t>(phase)) {
            return frame;
        }
    }
    throw std::runtime_error("oracle pack is missing a required phase");
}

float yaw_radians(quat rotation) {
    rotation = rotation / quat_length(rotation);
    const vec3 forward = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

float shortest_angle(float angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

interaction::Transform scene_alignment(
    interaction::Transform source_object,
    interaction::Transform target_object) {
    const float yaw = shortest_angle(
        yaw_radians(target_object.rotation) -
        yaw_radians(source_object.rotation));
    const quat rotation = quat_from_angle_axis(
        yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source = quat_mul_vec3(
        rotation, source_object.position);
    return {
        vec3(
            target_object.position.x - rotated_source.x,
            0.0F,
            target_object.position.z - rotated_source.z),
        rotation,
    };
}

interaction::Pose mapped_pose(
    const interaction::Database& database,
    int32_t frame,
    interaction::Transform scene_from_source) {
    interaction::Pose pose = interaction::pose_at_frame(database, frame);
    constexpr size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const interaction::Transform mapped_root = interaction::compose(
        scene_from_source,
        {pose.positions[root], pose.rotations[root]});
    pose.positions[root] = mapped_root.position;
    pose.rotations[root] = mapped_root.rotation;
    pose.velocities[root] = quat_mul_vec3(
        scene_from_source.rotation, pose.velocities[root]);
    pose.angular_velocities[root] = quat_mul_vec3(
        scene_from_source.rotation, pose.angular_velocities[root]);
    return pose;
}

interaction::InteractionTarget make_target(
    const interaction::Database& database,
    int32_t contact) {
    require(contact > database.range_starts.at(0U),
            "oracle pack has no pre-contact object sample");
    const interaction::Transform source_table = frame_transform(
        database.table_positions, database.table_rotations, 0U);
    const interaction::Transform source_object = frame_transform(
        database.object_positions,
        database.object_rotations,
        static_cast<size_t>(contact - 1));
    const interaction::Transform object_in_table = interaction::compose(
        interaction::inverse(source_table), source_object);

    interaction::InteractionTarget target{};
    target.handle = {1U, 1U};
    target.table_world = source_table;
    target.table_world.position.x = 0.0F;
    target.table_world.position.z = 3.0F;
    target.object_world = interaction::compose(
        target.table_world, object_in_table);
    target.table_size = read_vec3(database.table_sizes, 0U);
    target.object_profile_id = 1U;
    target.object_dimensions = read_vec3(database.object_dimensions, 0U);
    target.object_bounds = {
        vec3(), target.object_dimensions * 0.5F};
    target.state = interaction::ObjectState::Free;

    interaction::GraspAffordance affordance{};
    affordance.id = 1U;
    const uint8_t hand = database.active_hands.at(0U);
    require(hand <= 1U, "oracle clip 0 has an invalid active hand");
    affordance.hand = hand == 0U
        ? interaction::Hand::Left
        : interaction::Hand::Right;
    affordance.hand_in_object = frame_transform(
        database.grasp_positions_object,
        database.grasp_rotations_object,
        0U);
    affordance.approach_direction_object = read_vec3(
        database.approach_directions_object, 0U);
    affordance.clearance_radius = 0.04F;
    target.affordances.push_back(affordance);
    return target;
}

interaction::LocomotionSnapshot make_diagnostic_snapshot(
    const interaction::Database& database,
    int32_t reach,
    int32_t contact,
    const interaction::InteractionTarget& target) {
    const interaction::Transform source_object = frame_transform(
        database.object_positions,
        database.object_rotations,
        static_cast<size_t>(contact - 1));
    const interaction::Transform mapping = scene_alignment(
        source_object, target.object_world);
    interaction::LocomotionSnapshot snapshot{};
    snapshot.pose = mapped_pose(database, reach, mapping);
    constexpr std::array<int32_t, 3> future_offsets{8, 17, 25};
    const int32_t stop = database.range_stops.at(0U);
    for (size_t index = 0; index < future_offsets.size(); ++index) {
        const interaction::Pose future = mapped_pose(
            database,
            std::min(reach + future_offsets[index], stop - 1),
            mapping);
        snapshot.future_root_positions[index] =
            future.positions[g1_skeleton::Simulation];
        snapshot.future_root_rotations[index] =
            future.rotations[g1_skeleton::Simulation];
    }
    return snapshot;
}

bool near(float value, float expected) {
    return std::fabs(value - expected) <= kCostTolerance;
}

void require_ready(
    const interaction::PickEntryPreview& preview,
    float expected_cost,
    const char* message) {
    require(preview.path_feasible, message);
    require(preview.match_ready, message);
    require(preview.path_reason == interaction::Reason::None, message);
    require(preview.match_reason == interaction::Reason::None, message);
    require(preview.feasible_entry_frame == 114, message);
    require(preview.contact_frame == 139, message);
    require(preview.match_candidate.entry_frame == 114, message);
    require(preview.match_candidate.contact_frame == 139, message);
    require(near(preview.total_cost, expected_cost), message);
    require(near(preview.match_candidate.total_cost, expected_cost), message);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_pick_entry_oracle <pack-directory>\n";
        return 2;
    }
    try {
        const std::filesystem::path pack = argv[1];
        const interaction::Database database = interaction::load_database(
            pack / "interaction_database.bin");
        const interaction::Features features = interaction::load_features(
            pack / "interaction_features.bin");
        require(
            database.frame_count == features.frame_count,
            "oracle database/features frame count mismatch");
        const int32_t reach = first_phase_frame(
            database, interaction::Phase::Reach);
        const int32_t contact = first_phase_frame(
            database, interaction::Phase::Contact);
        require(reach == 114 && contact == 139,
                "oracle pack clip-0 phase frames changed");

        interaction::TargetRegistry registry{};
        const interaction::TargetHandle target_handle = registry.upsert(
            make_target(database, contact));
        const interaction::InteractionTarget* target = registry.find(
            target_handle);
        require(target != nullptr, "oracle target registration failed");
        const interaction::LocomotionSnapshot snapshot =
            make_diagnostic_snapshot(database, reach, contact, *target);
        const pick_entry_oracle::OracleRoots roots =
            pick_entry_oracle::make_oracle_roots(database, *target);
        interaction::InteractionRuntime runtime(
            database,
            features,
            registry,
            interaction::RuntimeConfig{});

        const interaction::PickEntryPreview reach_preview =
            runtime.preview_pick(
                snapshot, roots.reach, target_handle, 1U);
        const interaction::PickEntryPreview minus_preview =
            runtime.preview_pick(
                snapshot, roots.minus, target_handle, 1U);
        const interaction::PickEntryPreview plus_preview =
            runtime.preview_pick(
                snapshot, roots.plus, target_handle, 1U);

        require_ready(
            reach_preview, kExpectedReachCost, "R oracle mismatch");
        require(!minus_preview.path_feasible,
                "Minus oracle unexpectedly found a path");
        require(!minus_preview.match_ready,
                "Minus oracle unexpectedly became ready");
        require(
            minus_preview.path_reason == interaction::Reason::BlockedPath &&
            minus_preview.match_reason == interaction::Reason::BlockedPath,
            "Minus oracle lost BlockedPath");
        require_ready(
            plus_preview, kExpectedPlusCost, "Plus oracle mismatch");

        std::cout << std::fixed << std::setprecision(6)
                  << "R accepted entry="
                  << reach_preview.match_candidate.entry_frame
                  << " contact="
                  << reach_preview.match_candidate.contact_frame
                  << " cost=" << reach_preview.total_cost << '\n'
                  << "Minus rejected path_reason=BlockedPath\n"
                  << "Plus accepted entry="
                  << plus_preview.match_candidate.entry_frame
                  << " contact="
                  << plus_preview.match_candidate.contact_frame
                  << " cost=" << plus_preview.total_cost << '\n';
    } catch (const std::exception& error) {
        std::cerr << "pick-entry oracle: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
