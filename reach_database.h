#pragma once

#include "interaction_database.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace reach {

enum class Hand : uint8_t { Left = 0U, Right = 1U };
enum class Augmentation : uint8_t { Captured = 0U, Mirrored = 1U };

struct Database {
    uint32_t version = 0U;
    uint32_t endian_marker = 0U;
    uint32_t fps_numerator = 0U;
    uint32_t fps_denominator = 0U;
    uint32_t frame_count = 0U;
    uint32_t bone_count = 0U;
    uint32_t clip_count = 0U;
    uint32_t source_count = 0U;
    std::vector<int32_t> parents;
    std::vector<int32_t> range_starts;
    std::vector<int32_t> range_stops;
    std::vector<int32_t> contact_frames;
    std::vector<float> positions;
    std::vector<float> velocities;
    std::vector<float> rotations;
    std::vector<float> angular_velocities;
    std::vector<uint8_t> foot_contacts;
    std::vector<uint8_t> active_hands;
    std::vector<uint8_t> augmentations;
    std::vector<uint32_t> source_indices;
    std::vector<int32_t> original_indices;
    std::vector<float> endpoint_positions;
    std::vector<float> endpoint_rotations;
    std::vector<float> approach_directions;
    std::vector<int32_t> source_frames;
    std::vector<std::string> source_names;
};

struct Features {
    uint32_t version = 0U;
    uint32_t endian_marker = 0U;
    uint32_t clip_count = 0U;
    uint32_t dimension = 0U;
    std::vector<float> values;
};

struct Pack {
    Database database;
    Features features;
};

Database load_database(const std::filesystem::path& path);
Features load_features(const std::filesystem::path& path);
Pack load_pack(const std::filesystem::path& directory);

}  // namespace reach
