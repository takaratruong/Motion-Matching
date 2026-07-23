#include "reach_database.h"

#include "g1_skeleton.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <string>
#include <unordered_set>

namespace reach {
namespace {

constexpr std::array<char, 8> kDatabaseMagic = {
    'G','1','R','C','H','D','2','\0'};
constexpr std::array<char, 8> kFeatureMagic = {
    'G','1','R','C','H','F','2','\0'};
constexpr uint32_t kVersion = 2U;
constexpr uint32_t kEndian = 0x01020304U;
constexpr uint32_t kFpsNumerator = 25U;
constexpr uint32_t kFpsDenominator = 1U;
constexpr uint32_t kBones = 31U;
constexpr uint32_t kFeatureDimension = 10U;

void require_magic(
    std::istream& input,
    const std::array<char, 8>& expected,
    const char* label) {
    const auto actual = interaction::detail::read_magic(input, label);
    if (actual != expected) {
        throw interaction::FormatError(std::string("invalid ") + label);
    }
}

size_t product(
    std::initializer_list<size_t> dimensions,
    const char* label) {
    return interaction::detail::checked_product(dimensions, label);
}

template<class T>
std::vector<T> read_values(
    std::istream& input,
    size_t input_size,
    size_t count,
    const char* label) {
    return interaction::detail::read_guarded_vector<T>(
        input, input_size, count, label);
}

void require_finite(const std::vector<float>& values, const char* label) {
    if (!std::all_of(values.begin(), values.end(), [](float value) {
            return std::isfinite(value);
        })) {
        throw interaction::FormatError(
            std::string("non-finite reach ") + label);
    }
}

float squared_norm4(const std::vector<float>& values, size_t offset) {
    return values.at(offset) * values.at(offset) +
        values.at(offset + 1U) * values.at(offset + 1U) +
        values.at(offset + 2U) * values.at(offset + 2U) +
        values.at(offset + 3U) * values.at(offset + 3U);
}

float squared_norm3(const std::vector<float>& values, size_t offset) {
    return values.at(offset) * values.at(offset) +
        values.at(offset + 1U) * values.at(offset + 1U) +
        values.at(offset + 2U) * values.at(offset + 2U);
}

void validate_database(const Database& value) {
    if (value.version != kVersion || value.endian_marker != kEndian ||
        value.fps_numerator != kFpsNumerator ||
        value.fps_denominator != kFpsDenominator ||
        value.bone_count != kBones || value.frame_count == 0U ||
        value.clip_count == 0U || value.source_count == 0U) {
        throw interaction::FormatError("invalid reach database header");
    }
    if (value.parents.size() != g1_skeleton::kParents.size() ||
        !std::equal(
            value.parents.begin(), value.parents.end(),
            g1_skeleton::kParents.begin())) {
        throw interaction::FormatError("reach skeleton parents mismatch");
    }
    const size_t clips = value.clip_count;
    const size_t frames = value.frame_count;
    if (value.range_starts.size() != clips ||
        value.range_stops.size() != clips ||
        value.contact_frames.size() != clips ||
        value.active_hands.size() != clips ||
        value.augmentations.size() != clips ||
        value.source_indices.size() != clips ||
        value.original_indices.size() != clips ||
        value.endpoint_positions.size() != clips * 3U ||
        value.endpoint_rotations.size() != clips * 4U ||
        value.approach_directions.size() != clips * 3U ||
        value.source_frames.size() != frames ||
        value.source_names.size() != value.source_count) {
        throw interaction::FormatError("reach database metadata count mismatch");
    }
    if (value.range_starts.front() != 0 ||
        value.range_stops.back() != static_cast<int32_t>(frames)) {
        throw interaction::FormatError("reach ranges do not cover frames");
    }
    for (size_t clip = 0U; clip < clips; ++clip) {
        if (value.range_starts[clip] < 0 ||
            value.range_starts[clip] >= value.range_stops[clip] ||
            (clip > 0U &&
             value.range_starts[clip] != value.range_stops[clip - 1U])) {
            throw interaction::FormatError("invalid reach range");
        }
        if (value.contact_frames[clip] < value.range_starts[clip] ||
            value.contact_frames[clip] >= value.range_stops[clip]) {
            throw interaction::FormatError("invalid reach contact frame");
        }
        if (value.active_hands[clip] > 1U ||
            value.augmentations[clip] > 1U ||
            value.source_indices[clip] >= value.source_count) {
            throw interaction::FormatError("invalid reach clip metadata");
        }
        const int32_t original = value.original_indices[clip];
        if (value.augmentations[clip] ==
            static_cast<uint8_t>(Augmentation::Captured)) {
            if (original != -1) {
                throw interaction::FormatError(
                    "captured reach has original index");
            }
        } else if (original < 0 ||
                   static_cast<size_t>(original) >= clips ||
                   value.augmentations[static_cast<size_t>(original)] !=
                       static_cast<uint8_t>(Augmentation::Captured)) {
            throw interaction::FormatError(
                "mirrored reach has invalid original index");
        }
    }
    std::unordered_set<std::string> source_names;
    for (const std::string& name : value.source_names) {
        if (name.empty() || !source_names.insert(name).second) {
            throw interaction::FormatError("invalid reach source name");
        }
    }
    for (const auto* entry : {
             &value.positions, &value.velocities, &value.rotations,
             &value.angular_velocities, &value.endpoint_positions,
             &value.endpoint_rotations, &value.approach_directions}) {
        require_finite(*entry, "float channel");
    }
    if (value.positions.size() != frames * kBones * 3U ||
        value.velocities.size() != frames * kBones * 3U ||
        value.rotations.size() != frames * kBones * 4U ||
        value.angular_velocities.size() != frames * kBones * 3U ||
        value.foot_contacts.size() != frames * 2U) {
        throw interaction::FormatError("reach pose channel count mismatch");
    }
    for (size_t offset = 0U; offset < value.rotations.size(); offset += 4U) {
        if (std::abs(squared_norm4(value.rotations, offset) - 1.0F) > 2.0e-4F) {
            throw interaction::FormatError("reach pose quaternion norm error");
        }
    }
    for (size_t clip = 0U; clip < clips; ++clip) {
        if (std::abs(squared_norm4(
                value.endpoint_rotations, clip * 4U) - 1.0F) > 2.0e-4F) {
            throw interaction::FormatError(
                "reach endpoint quaternion norm error");
        }
        if (std::abs(squared_norm3(
                value.approach_directions, clip * 3U) - 1.0F) > 2.0e-4F) {
            throw interaction::FormatError(
                "reach approach direction norm error");
        }
    }
}

void validate_features(const Features& value, uint32_t expected_clips) {
    if (value.version != kVersion || value.endian_marker != kEndian ||
        value.clip_count != expected_clips ||
        value.dimension != kFeatureDimension ||
        value.values.size() !=
            static_cast<size_t>(expected_clips) * kFeatureDimension) {
        throw interaction::FormatError("invalid reach feature header");
    }
    require_finite(value.values, "features");
}

}  // namespace

Database load_database(const std::filesystem::path& path) {
    interaction::detail::require_little_endian_host();
    const size_t input_size = interaction::detail::file_size(path);
    std::ifstream input = interaction::detail::open_binary(path);
    require_magic(input, kDatabaseMagic, "reach database magic");
    Database result{};
    result.version = interaction::read_scalar<uint32_t>(input, "reach version");
    result.endian_marker = interaction::read_scalar<uint32_t>(input, "reach endian");
    result.fps_numerator = interaction::read_scalar<uint32_t>(input, "reach fps numerator");
    result.fps_denominator = interaction::read_scalar<uint32_t>(input, "reach fps denominator");
    result.frame_count = interaction::read_scalar<uint32_t>(input, "reach frames");
    result.bone_count = interaction::read_scalar<uint32_t>(input, "reach bones");
    result.clip_count = interaction::read_scalar<uint32_t>(input, "reach clips");
    result.source_count = interaction::read_scalar<uint32_t>(input, "reach sources");
    if (result.bone_count != kBones || result.frame_count == 0U ||
        result.clip_count == 0U || result.source_count == 0U) {
        throw interaction::FormatError("invalid reach database counts");
    }
    const size_t frames = result.frame_count;
    const size_t clips = result.clip_count;
    result.parents = read_values<int32_t>(input, input_size, kBones, "reach parents");
    result.range_starts = read_values<int32_t>(input, input_size, clips, "reach starts");
    result.range_stops = read_values<int32_t>(input, input_size, clips, "reach stops");
    result.contact_frames = read_values<int32_t>(
        input, input_size, clips, "reach contact frames");
    result.positions = read_values<float>(input, input_size,
        product({frames, kBones, 3U}, "reach positions"), "reach positions");
    result.velocities = read_values<float>(input, input_size,
        product({frames, kBones, 3U}, "reach velocities"), "reach velocities");
    result.rotations = read_values<float>(input, input_size,
        product({frames, kBones, 4U}, "reach rotations"), "reach rotations");
    result.angular_velocities = read_values<float>(input, input_size,
        product({frames, kBones, 3U}, "reach angular velocities"),
        "reach angular velocities");
    result.foot_contacts = read_values<uint8_t>(
        input, input_size, product({frames, 2U}, "reach contacts"),
        "reach contacts");
    result.active_hands = read_values<uint8_t>(input, input_size, clips, "reach hands");
    result.augmentations = read_values<uint8_t>(input, input_size, clips, "reach augmentations");
    result.source_indices = read_values<uint32_t>(input, input_size, clips, "reach source indices");
    result.original_indices = read_values<int32_t>(input, input_size, clips, "reach original indices");
    result.endpoint_positions = read_values<float>(input, input_size,
        product({clips, 3U}, "reach endpoints"), "reach endpoints");
    result.endpoint_rotations = read_values<float>(input, input_size,
        product({clips, 4U}, "reach endpoint rotations"),
        "reach endpoint rotations");
    result.approach_directions = read_values<float>(input, input_size,
        product({clips, 3U}, "reach approaches"), "reach approaches");
    result.source_frames = read_values<int32_t>(input, input_size, frames, "reach source frames");
    result.source_names.reserve(result.source_count);
    for (uint32_t source = 0U; source < result.source_count; ++source) {
        const uint32_t length = interaction::read_scalar<uint32_t>(
            input, "reach source name length");
        if (length == 0U || length > 4096U) {
            throw interaction::FormatError("invalid reach source name length");
        }
        const std::vector<char> bytes = read_values<char>(
            input, input_size, length, "reach source name");
        result.source_names.emplace_back(bytes.begin(), bytes.end());
    }
    interaction::detail::require_eof(input, "reach database");
    validate_database(result);
    return result;
}

Features load_features(const std::filesystem::path& path) {
    interaction::detail::require_little_endian_host();
    const size_t input_size = interaction::detail::file_size(path);
    std::ifstream input = interaction::detail::open_binary(path);
    require_magic(input, kFeatureMagic, "reach feature magic");
    Features result{};
    result.version = interaction::read_scalar<uint32_t>(input, "reach feature version");
    result.endian_marker = interaction::read_scalar<uint32_t>(input, "reach feature endian");
    result.clip_count = interaction::read_scalar<uint32_t>(input, "reach feature clips");
    result.dimension = interaction::read_scalar<uint32_t>(input, "reach feature dimension");
    if (result.clip_count == 0U || result.dimension != kFeatureDimension) {
        throw interaction::FormatError("invalid reach feature counts");
    }
    result.values = read_values<float>(input, input_size,
        product({result.clip_count, result.dimension}, "reach features"),
        "reach features");
    interaction::detail::require_eof(input, "reach features");
    validate_features(result, result.clip_count);
    return result;
}

Pack load_pack(const std::filesystem::path& directory) {
    Pack result{
        load_database(directory / "reach_database.bin"),
        load_features(directory / "reach_features.bin"),
    };
    validate_features(result.features, result.database.clip_count);
    for (size_t clip = 0U; clip < result.database.clip_count; ++clip) {
        const size_t feature = clip * kFeatureDimension;
        for (size_t component = 0U; component < 3U; ++component) {
            if (result.features.values[feature + component] !=
                    result.database.endpoint_positions[clip * 3U + component] ||
                result.features.values[feature + 3U + component] !=
                    result.database.approach_directions[clip * 3U + component]) {
                throw interaction::FormatError(
                    "reach features do not match database");
            }
        }
        for (size_t component = 0U; component < 4U; ++component) {
            if (result.features.values[feature + 6U + component] !=
                result.database.endpoint_rotations[clip * 4U + component]) {
                throw interaction::FormatError(
                    "reach features do not match database");
            }
        }
    }
    return result;
}

}  // namespace reach
