#include "interaction_features.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

template<class Integer>
Integer parse_nonnegative(std::string_view text, std::string_view label) {
    Integer value{};
    const char* const first = text.data();
    const char* const last = first + text.size();
    const auto result = std::from_chars(first, last, value);
    if (text.empty() || result.ec != std::errc() || result.ptr != last) {
        throw std::invalid_argument("invalid " + std::string(label));
    }
    return value;
}

vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 3U;
    return vec3(
        values.at(offset), values.at(offset + 1U), values.at(offset + 2U));
}

quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 4U;
    return quat(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U),
        values.at(offset + 3U));
}

interaction::QueryInput reconstruct_query(
    const interaction::Database& database,
    uint32_t clip,
    int32_t frame) {
    interaction::QueryInput input{};
    input.locomotion.pose = interaction::pose_at_frame(database, frame);

    constexpr std::array<int32_t, 3> future_offsets = {8, 17, 25};
    const int32_t stop = database.range_stops.at(clip);
    for (size_t sample = 0; sample < future_offsets.size(); ++sample) {
        const int32_t future = std::min(frame + future_offsets[sample], stop - 1);
        const interaction::Pose pose = interaction::pose_at_frame(database, future);
        input.locomotion.future_root_positions[sample] =
            pose.positions[g1_skeleton::Simulation];
        input.locomotion.future_root_rotations[sample] =
            pose.rotations[g1_skeleton::Simulation];
    }

    const size_t frame_index = static_cast<size_t>(frame);
    const size_t clip_index = static_cast<size_t>(clip);
    const vec3 object_position =
        read_vec3(database.object_positions, frame_index);
    const quat object_rotation =
        read_quat(database.object_rotations, frame_index);
    const vec3 grasp_position_object =
        read_vec3(database.grasp_positions_object, clip_index);
    const quat grasp_rotation_object =
        read_quat(database.grasp_rotations_object, clip_index);
    const vec3 grasp_offset_world =
        quat_mul_vec3(object_rotation, grasp_position_object);
    input.grasp_world = {
        object_position + grasp_offset_world,
        quat_mul(object_rotation, grasp_rotation_object),
    };

    const vec3 object_angular_velocity =
        read_vec3(database.object_angular_velocities, frame_index);
    input.grasp_linear_velocity =
        read_vec3(database.object_velocities, frame_index) +
        cross(object_angular_velocity, grasp_offset_world);
    input.grasp_angular_velocity = object_angular_velocity;
    input.table_world = {
        read_vec3(database.table_positions, clip_index),
        read_quat(database.table_rotations, clip_index),
    };
    input.table_size = read_vec3(database.table_sizes, clip_index);
    input.approach_direction_object =
        read_vec3(database.approach_directions_object, clip_index);
    input.object_dimensions =
        read_vec3(database.object_dimensions, clip_index);
    input.hand = database.active_hands.at(clip_index) == 0U
        ? interaction::Hand::Left
        : interaction::Hand::Right;
    return input;
}

void print_json(double maximum_error) {
    std::cout << std::setprecision(9)
              << "{\"dimension\":" << interaction::kFeatureDimension
              << ",\"max_abs_error\":" << maximum_error
              << ",\"groups\":[";
    for (size_t group = 0;
         group < interaction::kFeatureGroupStarts.size();
         ++group) {
        if (group != 0U) std::cout << ',';
        std::cout << '[' << interaction::kFeatureGroupStarts[group] << ','
                  << interaction::kFeatureGroupStops[group] << ']';
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 || std::string_view(argv[4]) != "--json") {
        std::cerr
            << "usage: interaction_query_probe <pack> <clip> <frame> --json\n";
        return 2;
    }

    try {
        const std::filesystem::path pack = argv[1];
        const uint32_t clip =
            parse_nonnegative<uint32_t>(argv[2], "clip");
        const int32_t frame =
            parse_nonnegative<int32_t>(argv[3], "frame");
        const interaction::Database database = interaction::load_database(
            pack / "interaction_database.bin");
        const interaction::Features features = interaction::load_features(
            pack / "interaction_features.bin");
        if (database.frame_count != features.frame_count) {
            throw interaction::FormatError(
                "feature/database frame count mismatch");
        }
        if (clip >= database.clip_count) {
            throw std::out_of_range("clip outside database");
        }
        const int32_t start = database.range_starts.at(clip);
        const int32_t stop = database.range_stops.at(clip);
        if (frame < start || frame >= stop) {
            throw std::out_of_range("frame outside clip range");
        }

        const interaction::RawQuery raw = interaction::build_raw_query(
            reconstruct_query(database, clip, frame));
        const interaction::NormalizedQuery normalized =
            interaction::normalize_query(raw, features);
        double maximum_error = 0.0;
        const size_t row =
            static_cast<size_t>(frame) * interaction::kFeatureDimension;
        for (size_t dimension = 0;
             dimension < interaction::kFeatureDimension;
             ++dimension) {
            maximum_error = std::max(
                maximum_error,
                std::abs(
                    static_cast<double>(normalized[dimension]) -
                    static_cast<double>(features.values.at(row + dimension))));
        }
        print_json(maximum_error);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interaction_query_probe: " << error.what() << '\n';
        return 1;
    }
}
