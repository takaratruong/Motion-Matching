#include "reach_motion.h"

#include <stdexcept>

namespace reach {
namespace {

vec3 read_vec3(const std::vector<float>& values, size_t offset) {
    return vec3(values.at(offset), values.at(offset + 1U), values.at(offset + 2U));
}

quat read_quat(const std::vector<float>& values, size_t offset) {
    return quat(
        values.at(offset), values.at(offset + 1U),
        values.at(offset + 2U), values.at(offset + 3U));
}

}  // namespace

interaction::Pose pose_at_frame(const Database& database, int32_t frame) {
    if (frame < 0 || static_cast<uint32_t>(frame) >= database.frame_count) {
        throw std::out_of_range("reach pose frame outside database");
    }
    const size_t source_frame = static_cast<size_t>(frame);
    interaction::Pose pose{};
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        const size_t vector_offset =
            (source_frame * g1_skeleton::BoneCount + bone) * 3U;
        const size_t rotation_offset =
            (source_frame * g1_skeleton::BoneCount + bone) * 4U;
        pose.positions[bone] = read_vec3(database.positions, vector_offset);
        pose.velocities[bone] = read_vec3(database.velocities, vector_offset);
        pose.rotations[bone] = read_quat(database.rotations, rotation_offset);
        pose.angular_velocities[bone] = read_vec3(
            database.angular_velocities, vector_offset);
    }
    pose.foot_contacts[0] = database.foot_contacts.at(source_frame * 2U);
    pose.foot_contacts[1] = database.foot_contacts.at(source_frame * 2U + 1U);
    return pose;
}

interaction::Transform endpoint_transform(
    const Database& database,
    size_t clip) {
    if (clip >= database.clip_count) {
        throw std::out_of_range("reach endpoint clip outside database");
    }
    return {
        read_vec3(database.endpoint_positions, clip * 3U),
        read_quat(database.endpoint_rotations, clip * 4U),
    };
}

vec3 approach_direction(const Database& database, size_t clip) {
    if (clip >= database.clip_count) {
        throw std::out_of_range("reach approach clip outside database");
    }
    return read_vec3(database.approach_directions, clip * 3U);
}

}  // namespace reach

