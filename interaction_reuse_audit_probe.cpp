#include "interaction_reuse_audit.h"
#include "interaction_trajectory_database.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = 3U * index;
    return vec3(
        values.at(offset), values.at(offset + 1U), values.at(offset + 2U));
}

quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = 4U * index;
    return quat_normalize(quat(
        values.at(offset), values.at(offset + 1U),
        values.at(offset + 2U), values.at(offset + 3U)));
}

struct CanonicalGrasp {
    size_t clip = 0U;
    interaction::Hand hand = interaction::Hand::Right;
    vec3 dimensions{};
    interaction::Transform grasp_in_object{};
    vec3 approach_in_object{1.0F, 0.0F, 0.0F};
    interaction::Transform source_object{};
    interaction::Transform table_world{};
    vec3 table_dimensions{};
};

CanonicalGrasp find_table_grasp(const interaction::Database& database) {
    constexpr uint8_t contact_phase = 2U;
    for (size_t clip = 0U; clip < database.clip_count; ++clip) {
        if (interaction::support_kind(database, clip) !=
            interaction::SupportKind::Table) {
            continue;
        }
        const int32_t start = database.range_starts.at(clip);
        const int32_t stop = database.range_stops.at(clip);
        for (int32_t frame = start; frame < stop; ++frame) {
            if (database.phases.at(static_cast<size_t>(frame)) !=
                contact_phase) {
                continue;
            }
            const int32_t object_frame = std::max(start, frame - 1);
            return {
                clip,
                static_cast<interaction::Hand>(
                    database.active_hands.at(clip)),
                read_vec3(database.object_dimensions, clip),
                {
                    read_vec3(database.grasp_positions_object, clip),
                    read_quat(database.grasp_rotations_object, clip),
                },
                read_vec3(database.approach_directions_object, clip),
                {
                    read_vec3(
                        database.object_positions,
                        static_cast<size_t>(object_frame)),
                    read_quat(
                        database.object_rotations,
                        static_cast<size_t>(object_frame)),
                },
                {
                    read_vec3(database.table_positions, clip),
                    read_quat(database.table_rotations, clip),
                },
                read_vec3(database.table_sizes, clip),
            };
        }
    }
    throw std::runtime_error("interaction database has no table Contact clip");
}

interaction::HandTrajectoryQuery make_query(
    const CanonicalGrasp& canonical,
    const interaction::Transform& object_world) {
    const interaction::Transform grasp_world = interaction::compose(
        object_world, canonical.grasp_in_object);
    interaction::HandTrajectoryQuery query{};
    query.object_world = object_world;
    query.object_dimensions = canonical.dimensions;
    query.hand = canonical.hand;
    query.grasp_world_position = grasp_world.position;
    query.grasp_world_rotation = grasp_world.rotation;
    query.approach_world_direction = normalize(quat_mul_vec3(
        object_world.rotation, canonical.approach_in_object));
    query.orientation_mode =
        interaction::GraspOrientationMode::ApproachAxis;
    return query;
}

interaction::TrajectoryCollisionConfig viewer_collision_config() {
    interaction::TrajectoryCollisionConfig config{};
    config.joint_radius_m += 0.02F;
    config.limb_radius_m += 0.02F;
    config.torso_radius_m += 0.02F;
    return config;
}

void print_result(
    const std::string& scenario,
    const interaction::ReuseAuditResult& result) {
    const char* status = result.status ==
            interaction::ReuseAuditStatus::Complete
        ? "complete"
        : "incomplete";
    std::cout
        << "{\"scenario\":\"" << scenario
        << "\",\"status\":\"" << status
        << "\",\"elapsed_milliseconds\":"
        << result.elapsed_milliseconds
        << ",\"total\":" << result.counts.total
        << ",\"processed\":" << result.counts.processed
        << ",\"contact_accepted\":" << result.counts.contact_accepted
        << ",\"fully_shaped\":" << result.counts.fully_shaped
        << ",\"object_rejected\":" << result.counts.object_rejected
        << ",\"environment_rejected\":"
        << result.counts.environment_rejected
        << ",\"reusable\":" << result.counts.reusable
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: interaction_reuse_audit_probe <pack> "
                "<shelf|under-table>");
        }
        const std::string scenario = argv[2];
        if (scenario != "shelf" && scenario != "under-table") {
            throw std::invalid_argument("scenario must be shelf or under-table");
        }
        const std::filesystem::path pack = argv[1];
        const interaction::Database database =
            interaction::load_trajectory_database(
                pack / "interaction_database.bin");
        const CanonicalGrasp canonical = find_table_grasp(database);
        const vec3 scene_offset(
            -canonical.table_world.position.x,
            0.0F,
            -canonical.table_world.position.z);
        interaction::Transform table_world = canonical.table_world;
        table_world.position = table_world.position + scene_offset;
        const interaction::EnvironmentGeometry environment =
            interaction::make_coverage_environment(
                table_world, canonical.table_dimensions);

        interaction::Transform object_world = canonical.source_object;
        object_world.position = object_world.position + scene_offset;
        if (scenario == "shelf") {
            if (environment.boxes.size() <= 5U) {
                throw std::runtime_error("coverage shelf board is unavailable");
            }
            const interaction::OrientedBox& shelf = environment.boxes[5U];
            object_world.position = shelf.world.position;
            object_world.position.y +=
                0.5F * (shelf.dimensions.y + canonical.dimensions.y);
        } else {
            object_world.position = vec3(
                table_world.position.x,
                0.5F * canonical.dimensions.y,
                table_world.position.z);
        }

        const interaction::HandTrajectoryQuery query = make_query(
            canonical, object_world);
        interaction::ReuseAuditConfig config{};
        config.worker_count = 4U;
        config.deadline_milliseconds = 30000U;
        const interaction::ReuseAuditResult result =
            interaction::audit_reusable_hand_trajectories(
                database,
                query,
                {object_world, canonical.dimensions},
                environment,
                viewer_collision_config(),
                config);
        print_result(scenario, result);
        return result.status == interaction::ReuseAuditStatus::Complete
            ? 0
            : 3;
    } catch (const std::exception& error) {
        std::cerr << "interaction reuse audit probe: " << error.what() << '\n';
        return 2;
    }
}
