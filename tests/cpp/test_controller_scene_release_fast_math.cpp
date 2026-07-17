#include "interaction_controller_adapter.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

interaction::Database frozen_release_scene_database() {
    interaction::Database database{};
    database.frame_count = 4U;
    database.bone_count = static_cast<uint32_t>(g1_skeleton::BoneCount);
    database.clip_count = 1U;
    database.hand_dof_count = 14U;
    database.range_starts = {0};
    database.range_stops = {4};
    database.phases = {
        static_cast<uint8_t>(interaction::Phase::Approach),
        static_cast<uint8_t>(interaction::Phase::Reach),
        static_cast<uint8_t>(interaction::Phase::Contact),
        static_cast<uint8_t>(interaction::Phase::Lift)};
    database.active_hands = {
        static_cast<uint8_t>(interaction::Hand::Right)};

    const size_t pose_samples =
        static_cast<size_t>(database.frame_count) * g1_skeleton::BoneCount;
    database.positions.assign(pose_samples * 3U, 0.0F);
    database.velocities.assign(pose_samples * 3U, 0.0F);
    database.rotations.assign(pose_samples * 4U, 0.0F);
    database.angular_velocities.assign(pose_samples * 3U, 0.0F);
    for (size_t sample = 0; sample < pose_samples; ++sample) {
        database.rotations[sample * 4U] = 1.0F;
    }
    database.hand_dof.assign(4U * 14U, 0.0F);
    database.hand_dof_velocities.assign(4U * 14U, 0.0F);
    database.foot_contacts.assign(4U * 2U, 0U);

    database.object_positions.assign(4U * 3U, 0.0F);
    database.object_rotations.assign(4U * 4U, 0.0F);
    for (size_t frame = 0; frame < 4U; ++frame) {
        database.object_positions[frame * 3U] = 0.142707825F;
        database.object_positions[frame * 3U + 1U] = 0.797914684F;
        database.object_positions[frame * 3U + 2U] = -0.183830261F;
        database.object_rotations[frame * 4U] = -0.090692319F;
        database.object_rotations[frame * 4U + 1U] = 0.643643677F;
        database.object_rotations[frame * 4U + 2U] = 0.287561417F;
        database.object_rotations[frame * 4U + 3U] = -0.703424633F;
    }

    database.table_positions = {
        0.0F, 0.679573536F, -1.11022302e-16F};
    database.table_rotations = {1.0F, 0.0F, 0.0F, 0.0F};
    database.table_sizes = {2.0F, 0.0399999991F, 0.600000024F};
    database.object_dimensions = {
        0.0461015404F, 0.190876126F, 0.109788500F};
    database.approach_directions_object = {0.0F, 0.0F, 1.0F};
    return database;
}

}  // namespace

int main() {
    try {
        const interaction::Database database =
            frozen_release_scene_database();
        const interaction::InteractionTarget target =
            interaction::make_controller_demo_target(database);
        const interaction::PlacementSurface destination =
            interaction::make_controller_demo_destination_surface(
                database, target);
        require(
            destination.affordances.size() == 1U,
            "release scene did not author exactly one destination affordance");
        const interaction::PlacementFit fit =
            interaction::evaluate_placement_fit(
                destination,
                destination.affordances.front(),
                target.object_bounds);
        require(fit.accepted, "release scene final placement fit was rejected");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
