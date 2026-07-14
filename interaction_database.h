#pragma once

#if defined(__has_include)
#if __has_include(<bit>)
#include <bit>
#endif
#endif

#include "g1_skeleton.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <istream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace interaction {

class FormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template<class T>
T read_scalar(std::istream& input, std::string_view label) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw FormatError("truncated " + std::string(label));
    return value;
}

template<class T>
std::vector<T> read_vector(
    std::istream& input, size_t count, std::string_view label) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
        throw FormatError("overflow " + std::string(label));
    std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) throw FormatError("truncated " + std::string(label));
    return values;
}

struct Database {
    uint32_t version = 0;
    uint32_t endian_marker = 0;
    uint32_t fps_numerator = 0;
    uint32_t fps_denominator = 0;
    uint32_t frame_count = 0;
    uint32_t bone_count = 0;
    uint32_t clip_count = 0;
    uint32_t hand_dof_count = 0;

    std::vector<int32_t> parents;
    std::vector<int32_t> range_starts;
    std::vector<int32_t> range_stops;
    std::vector<float> positions;
    std::vector<float> velocities;
    std::vector<float> rotations;
    std::vector<float> angular_velocities;
    std::vector<uint8_t> foot_contacts;
    std::vector<uint8_t> hand_contacts;
    std::vector<float> hand_dof;
    std::vector<float> hand_dof_velocities;
    std::vector<uint8_t> phases;
    std::vector<uint8_t> active_hands;
    std::vector<float> time_to_contact;
    std::vector<float> object_positions;
    std::vector<float> object_rotations;
    std::vector<float> object_velocities;
    std::vector<float> object_angular_velocities;
    std::vector<float> table_positions;
    std::vector<float> table_rotations;
    std::vector<float> table_sizes;
    std::vector<float> object_dimensions;
    std::vector<float> grasp_positions_object;
    std::vector<float> grasp_rotations_object;
    std::vector<float> approach_directions_object;
    std::vector<int32_t> source_frames;
};

struct Features {
    uint32_t version = 0;
    uint32_t endian_marker = 0;
    uint32_t frame_count = 0;
    uint32_t feature_count = 0;
    uint32_t dimension = 0;
    uint32_t group_count = 0;

    std::vector<uint32_t> group_starts;
    std::vector<uint32_t> group_stops;
    std::vector<float> offsets;
    std::vector<float> scales;
    std::vector<float> values;
};

namespace detail {

inline constexpr uint32_t kVersion = 1;
inline constexpr uint32_t kEndianMarker = 0x01020304U;
inline constexpr uint32_t kFpsNumerator = 25;
inline constexpr uint32_t kFpsDenominator = 1;
inline constexpr uint32_t kBoneCount = 31;
inline constexpr uint32_t kHandDofCount = 14;
inline constexpr uint32_t kFeatureCount = 71;
inline constexpr uint32_t kFeatureGroupCount = 5;
inline constexpr double kFloatTolerance = 1.0e-4;
inline constexpr size_t kHoldContactSamples = 5;

inline void require_little_endian_host() {
#if defined(__cpp_lib_endian) && __cpp_lib_endian >= 201907L
    if constexpr (std::endian::native != std::endian::little) {
        throw FormatError("unsupported host endian; little-endian required");
    }
#else
    const uint32_t marker = kEndianMarker;
    const auto* first_byte = reinterpret_cast<const uint8_t*>(&marker);
    if (*first_byte != 0x04U) {
        throw FormatError("unsupported host endian; little-endian required");
    }
#endif
}

inline size_t checked_multiply(
    size_t left,
    size_t right,
    std::string_view label) {
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
        throw FormatError("overflow " + std::string(label));
    }
    return left * right;
}

inline size_t checked_product(
    std::initializer_list<size_t> dimensions,
    std::string_view label) {
    size_t result = 1;
    for (size_t dimension : dimensions) {
        result = checked_multiply(result, dimension, label);
    }
    return result;
}

inline size_t file_size(const std::filesystem::path& path) {
    std::error_code error;
    const uintmax_t result = std::filesystem::file_size(path, error);
    if (error) {
        throw FormatError(
            "cannot determine file size for " + path.string() + ": " +
            error.message());
    }
    if (result > std::numeric_limits<size_t>::max()) {
        throw FormatError("file size overflow for " + path.string());
    }
    return static_cast<size_t>(result);
}

inline std::ifstream open_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw FormatError("cannot open " + path.string());
    }
    return input;
}

inline std::array<char, 8> read_magic(
    std::istream& input,
    std::string_view label) {
    std::array<char, 8> value{};
    input.read(value.data(), static_cast<std::streamsize>(value.size()));
    if (!input) {
        throw FormatError("truncated " + std::string(label));
    }
    return value;
}

inline bool magic_equals(
    const std::array<char, 8>& actual,
    std::string_view expected) {
    return expected.size() == actual.size() &&
           std::equal(actual.begin(), actual.end(), expected.begin());
}

template<class T>
std::vector<T> read_guarded_vector(
    std::istream& input,
    size_t input_size,
    size_t count,
    std::string_view label) {
    const size_t byte_count = checked_multiply(count, sizeof(T), label);
    const std::streampos position = input.tellg();
    if (position < 0) {
        throw FormatError("cannot determine offset for " + std::string(label));
    }
    const auto offset = static_cast<uintmax_t>(position);
    if (offset > input_size || byte_count > input_size - offset) {
        throw FormatError("truncated " + std::string(label));
    }
    return read_vector<T>(input, count, label);
}

inline void require_eof(std::istream& input, std::string_view label) {
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw FormatError(std::string(label) + " has trailing bytes");
    }
    if (!input.eof()) {
        throw FormatError("could not validate " + std::string(label) + " EOF");
    }
}

inline void require_header_count(
    uint32_t count,
    size_t input_size,
    std::string_view label) {
    if (count == 0) {
        throw FormatError("invalid " + std::string(label) + " 0");
    }
    if (count > input_size) {
        throw FormatError(
            "invalid " + std::string(label) + " exceeds file size");
    }
}

inline void require_finite(
    const std::vector<float>& values,
    std::string_view label) {
    if (!std::all_of(values.begin(), values.end(), [](float value) {
            return std::isfinite(value);
        })) {
        throw FormatError(std::string(label) + " contains non-finite values");
    }
}

inline void require_binary(
    const std::vector<uint8_t>& values,
    std::string_view label) {
    if (!std::all_of(values.begin(), values.end(), [](uint8_t value) {
            return value <= 1U;
        })) {
        throw FormatError(std::string(label) + " must contain only 0 or 1");
    }
}

inline void require_unit_quaternions(
    const std::vector<float>& values,
    std::string_view label) {
    if (values.size() % 4U != 0U) {
        throw FormatError("invalid " + std::string(label) + " count");
    }
    for (size_t index = 0; index < values.size(); index += 4U) {
        double squared_norm = 0.0;
        for (size_t component = 0; component < 4U; ++component) {
            const double value = static_cast<double>(values[index + component]);
            squared_norm += value * value;
        }
        const double error = std::abs(std::sqrt(squared_norm) - 1.0);
        if (error > kFloatTolerance) {
            throw FormatError(
                std::string(label) + " must contain unit quaternions");
        }
    }
}

inline void validate_ranges(const Database& database) {
    if (database.range_starts.front() != 0) {
        throw FormatError("range_starts must begin at 0");
    }
    for (size_t clip = 0; clip < database.range_starts.size(); ++clip) {
        const int32_t start = database.range_starts[clip];
        const int32_t stop = database.range_stops[clip];
        if (start < 0) {
            throw FormatError("range_starts must be nonnegative");
        }
        if (stop <= start) {
            throw FormatError("range_stops must define nonempty ranges");
        }
        if (clip != 0U && start != database.range_stops[clip - 1U]) {
            throw FormatError("range_starts must provide contiguous coverage");
        }
    }
    if (database.range_stops.back() !=
        static_cast<int64_t>(database.frame_count)) {
        throw FormatError("range_stops must end at frame_count");
    }
}

inline void validate_phases_and_clip_semantics(const Database& database) {
    for (uint8_t phase : database.phases) {
        if (phase > 4U) {
            throw FormatError("phases must contain values from 0 through 4");
        }
    }

    for (size_t clip = 0; clip < database.range_starts.size(); ++clip) {
        const size_t start = static_cast<size_t>(database.range_starts[clip]);
        const size_t stop = static_cast<size_t>(database.range_stops[clip]);
        bool has_contact = false;
        bool has_lift = false;
        size_t hold_count = 0;
        size_t contacted_hold_count = 0;
        uint8_t previous_phase = database.phases[start];
        int32_t previous_source_frame = database.source_frames[start];
        float previous_time = database.time_to_contact[start];
        const uint8_t active_hand = database.active_hands[clip];

        if (previous_source_frame < 0) {
            throw FormatError("source_frames must be nonnegative");
        }
        for (size_t frame = start; frame < stop; ++frame) {
            const uint8_t phase = database.phases[frame];
            if (frame != start && phase < previous_phase) {
                throw FormatError("phases must be monotonic within each range");
            }
            previous_phase = phase;

            const int32_t source_frame = database.source_frames[frame];
            if (source_frame < 0 ||
                (frame != start && source_frame < previous_source_frame)) {
                throw FormatError(
                    "source_frames must be nonnegative and nondecreasing "
                    "within each range");
            }
            previous_source_frame = source_frame;

            const float time = database.time_to_contact[frame];
            if (time < -static_cast<float>(kFloatTolerance)) {
                throw FormatError("time_to_contact must be nonnegative");
            }
            if (frame != start &&
                static_cast<double>(time) - static_cast<double>(previous_time) >
                    kFloatTolerance) {
                throw FormatError("time_to_contact must be nonincreasing");
            }
            if (phase >= 2U && std::abs(static_cast<double>(time)) > kFloatTolerance) {
                throw FormatError("time_to_contact must be zero from CONTACT onward");
            }
            previous_time = time;

            const uint8_t contact = database.hand_contacts[frame * 2U + active_hand];
            if (phase == 2U) {
                has_contact = true;
                if (contact != 1U) {
                    throw FormatError(
                        "hand_contacts must contain active-hand CONTACT contact");
                }
            } else if (phase == 3U) {
                has_lift = true;
                if (contact != 1U) {
                    throw FormatError(
                        "hand_contacts must contain active-hand LIFT contact");
                }
            } else if (phase == 4U) {
                ++hold_count;
                if (hold_count <= kHoldContactSamples && contact == 1U) {
                    ++contacted_hold_count;
                }
            }
        }
        if (!has_contact || !has_lift || hold_count == 0U) {
            throw FormatError(
                "phases in each range must include CONTACT, LIFT, and HOLD");
        }
        if (hold_count < kHoldContactSamples) {
            throw FormatError("phases must include at least 5 HOLD samples per range");
        }
        if (contacted_hold_count != kHoldContactSamples) {
            throw FormatError(
                "hand_contacts must cover the first 5 HOLD samples");
        }

        const size_t approach = clip * 3U;
        const double x = database.approach_directions_object[approach];
        const double y = database.approach_directions_object[approach + 1U];
        const double z = database.approach_directions_object[approach + 2U];
        if (std::abs(y) > kFloatTolerance) {
            throw FormatError(
                "approach_directions_object must be horizontal");
        }
        const double norm = std::sqrt(x * x + y * y + z * z);
        if (std::abs(norm - 1.0) > kFloatTolerance) {
            throw FormatError(
                "approach_directions_object must contain unit directions");
        }
    }
}

inline void validate_database(const Database& database) {
    if (!std::equal(
            database.parents.begin(),
            database.parents.end(),
            g1_skeleton::kParents.begin())) {
        throw FormatError("parents must match the exact G1 parent array");
    }
    validate_ranges(database);

    require_finite(database.positions, "positions");
    require_finite(database.velocities, "velocities");
    require_finite(database.rotations, "rotations");
    require_finite(database.angular_velocities, "angular_velocities");
    require_finite(database.hand_dof, "hand_dof");
    require_finite(database.hand_dof_velocities, "hand_dof_velocities");
    require_finite(database.time_to_contact, "time_to_contact");
    require_finite(database.object_positions, "object_positions");
    require_finite(database.object_rotations, "object_rotations");
    require_finite(database.object_velocities, "object_velocities");
    require_finite(
        database.object_angular_velocities, "object_angular_velocities");
    require_finite(database.table_positions, "table_positions");
    require_finite(database.table_rotations, "table_rotations");
    require_finite(database.table_sizes, "table_sizes");
    require_finite(database.object_dimensions, "object_dimensions");
    require_finite(
        database.grasp_positions_object, "grasp_positions_object");
    require_finite(
        database.grasp_rotations_object, "grasp_rotations_object");
    require_finite(
        database.approach_directions_object,
        "approach_directions_object");

    require_binary(database.foot_contacts, "foot_contacts");
    require_binary(database.hand_contacts, "hand_contacts");
    require_binary(database.active_hands, "active_hands");
    validate_phases_and_clip_semantics(database);

    require_unit_quaternions(database.rotations, "rotations");
    require_unit_quaternions(database.object_rotations, "object_rotations");
    require_unit_quaternions(database.table_rotations, "table_rotations");
    require_unit_quaternions(
        database.grasp_rotations_object, "grasp_rotations_object");

    if (!std::all_of(
            database.table_sizes.begin(),
            database.table_sizes.end(),
            [](float value) { return value > 0.0F; })) {
        throw FormatError("table_sizes must be positive");
    }
    if (!std::all_of(
            database.object_dimensions.begin(),
            database.object_dimensions.end(),
            [](float value) { return value > 0.0F; })) {
        throw FormatError("object_dimensions must be positive");
    }
}

inline void validate_features(const Features& features) {
    constexpr std::array<uint32_t, kFeatureGroupCount> expected_starts = {
        0, 33, 45, 57, 65};
    constexpr std::array<uint32_t, kFeatureGroupCount> expected_stops = {
        33, 45, 57, 65, 71};
    if (!std::equal(
            features.group_starts.begin(),
            features.group_starts.end(),
            expected_starts.begin()) ||
        !std::equal(
            features.group_stops.begin(),
            features.group_stops.end(),
            expected_stops.begin())) {
        throw FormatError(
            "invalid feature group ranges; expected [0,33), [33,45), "
            "[45,57), [57,65), [65,71)");
    }
    require_finite(features.offsets, "feature offsets");
    require_finite(features.scales, "feature scales");
    require_finite(features.values, "feature values");
    if (!std::all_of(
            features.scales.begin(),
            features.scales.end(),
            [](float value) { return value > 0.0F; })) {
        throw FormatError("feature scales must be positive");
    }
}

}  // namespace detail

inline Database load_database(const std::filesystem::path& path) {
    detail::require_little_endian_host();
    const size_t input_size = detail::file_size(path);
    std::ifstream input = detail::open_binary(path);

    const std::array<char, 8> magic =
        detail::read_magic(input, "database magic");
    if (!detail::magic_equals(magic, "G1INTDB1")) {
        throw FormatError("invalid database magic");
    }

    Database database;
    database.version = read_scalar<uint32_t>(input, "database version");
    database.endian_marker =
        read_scalar<uint32_t>(input, "database endian marker");
    database.fps_numerator =
        read_scalar<uint32_t>(input, "database fps numerator");
    database.fps_denominator =
        read_scalar<uint32_t>(input, "database fps denominator");
    database.frame_count =
        read_scalar<uint32_t>(input, "database frame count");
    database.bone_count =
        read_scalar<uint32_t>(input, "database bone count");
    database.clip_count =
        read_scalar<uint32_t>(input, "database clip count");
    database.hand_dof_count =
        read_scalar<uint32_t>(input, "database hand dof count");

    if (database.version != detail::kVersion) {
        throw FormatError("invalid database version; expected 1");
    }
    if (database.endian_marker != detail::kEndianMarker) {
        throw FormatError("invalid database endian marker; expected 0x01020304");
    }
    if (database.fps_numerator != detail::kFpsNumerator ||
        database.fps_denominator != detail::kFpsDenominator) {
        throw FormatError("invalid database fps; expected 25/1");
    }
    detail::require_header_count(
        database.frame_count, input_size, "database frame count");
    if (database.bone_count != detail::kBoneCount) {
        throw FormatError("invalid database bone count; expected 31");
    }
    detail::require_header_count(
        database.clip_count, input_size, "database clip count");
    if (database.hand_dof_count != detail::kHandDofCount) {
        throw FormatError("invalid database hand dof count; expected 14");
    }

    const size_t frames = database.frame_count;
    const size_t bones = database.bone_count;
    const size_t clips = database.clip_count;
    const size_t hand_dofs = database.hand_dof_count;
    const size_t frame_bones =
        detail::checked_product({frames, bones}, "frame/bone dimensions");
    const size_t bone_vectors = detail::checked_product(
        {frame_bones, 3U}, "bone vector dimensions");
    const size_t bone_quaternions = detail::checked_product(
        {frame_bones, 4U}, "bone quaternion dimensions");
    const size_t frame_pairs =
        detail::checked_product({frames, 2U}, "contact dimensions");
    const size_t hand_values = detail::checked_product(
        {frames, hand_dofs}, "hand dof dimensions");
    const size_t frame_vectors =
        detail::checked_product({frames, 3U}, "frame vector dimensions");
    const size_t frame_quaternions = detail::checked_product(
        {frames, 4U}, "frame quaternion dimensions");
    const size_t clip_vectors =
        detail::checked_product({clips, 3U}, "clip vector dimensions");
    const size_t clip_quaternions = detail::checked_product(
        {clips, 4U}, "clip quaternion dimensions");

    database.parents = detail::read_guarded_vector<int32_t>(
        input, input_size, bones, "parents");
    database.range_starts = detail::read_guarded_vector<int32_t>(
        input, input_size, clips, "range_starts");
    database.range_stops = detail::read_guarded_vector<int32_t>(
        input, input_size, clips, "range_stops");
    database.positions = detail::read_guarded_vector<float>(
        input, input_size, bone_vectors, "positions");
    database.velocities = detail::read_guarded_vector<float>(
        input, input_size, bone_vectors, "velocities");
    database.rotations = detail::read_guarded_vector<float>(
        input, input_size, bone_quaternions, "rotations");
    database.angular_velocities = detail::read_guarded_vector<float>(
        input, input_size, bone_vectors, "angular_velocities");
    database.foot_contacts = detail::read_guarded_vector<uint8_t>(
        input, input_size, frame_pairs, "foot_contacts");
    database.hand_contacts = detail::read_guarded_vector<uint8_t>(
        input, input_size, frame_pairs, "hand_contacts");
    database.hand_dof = detail::read_guarded_vector<float>(
        input, input_size, hand_values, "hand_dof");
    database.hand_dof_velocities = detail::read_guarded_vector<float>(
        input, input_size, hand_values, "hand_dof_velocities");
    database.phases = detail::read_guarded_vector<uint8_t>(
        input, input_size, frames, "phases");
    database.active_hands = detail::read_guarded_vector<uint8_t>(
        input, input_size, clips, "active_hands");
    database.time_to_contact = detail::read_guarded_vector<float>(
        input, input_size, frames, "time_to_contact");
    database.object_positions = detail::read_guarded_vector<float>(
        input, input_size, frame_vectors, "object_positions");
    database.object_rotations = detail::read_guarded_vector<float>(
        input, input_size, frame_quaternions, "object_rotations");
    database.object_velocities = detail::read_guarded_vector<float>(
        input, input_size, frame_vectors, "object_velocities");
    database.object_angular_velocities = detail::read_guarded_vector<float>(
        input, input_size, frame_vectors, "object_angular_velocities");
    database.table_positions = detail::read_guarded_vector<float>(
        input, input_size, clip_vectors, "table_positions");
    database.table_rotations = detail::read_guarded_vector<float>(
        input, input_size, clip_quaternions, "table_rotations");
    database.table_sizes = detail::read_guarded_vector<float>(
        input, input_size, clip_vectors, "table_sizes");
    database.object_dimensions = detail::read_guarded_vector<float>(
        input, input_size, clip_vectors, "object_dimensions");
    database.grasp_positions_object = detail::read_guarded_vector<float>(
        input, input_size, clip_vectors, "grasp_positions_object");
    database.grasp_rotations_object = detail::read_guarded_vector<float>(
        input, input_size, clip_quaternions, "grasp_rotations_object");
    database.approach_directions_object = detail::read_guarded_vector<float>(
        input, input_size, clip_vectors, "approach_directions_object");
    database.source_frames = detail::read_guarded_vector<int32_t>(
        input, input_size, frames, "source_frames");

    detail::require_eof(input, "database");
    detail::validate_database(database);
    return database;
}

inline Features load_features(const std::filesystem::path& path) {
    detail::require_little_endian_host();
    const size_t input_size = detail::file_size(path);
    std::ifstream input = detail::open_binary(path);

    const std::array<char, 8> magic = detail::read_magic(input, "feature magic");
    if (!detail::magic_equals(magic, "G1INTFT1")) {
        throw FormatError("invalid feature magic");
    }

    Features features;
    features.version = read_scalar<uint32_t>(input, "feature version");
    features.endian_marker =
        read_scalar<uint32_t>(input, "feature endian marker");
    features.frame_count =
        read_scalar<uint32_t>(input, "feature frame count");
    features.feature_count =
        read_scalar<uint32_t>(input, "feature count");
    features.dimension = features.feature_count;
    features.group_count =
        read_scalar<uint32_t>(input, "feature group count");

    if (features.version != detail::kVersion) {
        throw FormatError("invalid feature version; expected 1");
    }
    if (features.endian_marker != detail::kEndianMarker) {
        throw FormatError("invalid feature endian marker; expected 0x01020304");
    }
    detail::require_header_count(
        features.frame_count, input_size, "feature frame count");
    if (features.feature_count != detail::kFeatureCount) {
        throw FormatError(
            "invalid feature count (feature dimension); expected 71");
    }
    if (features.group_count != detail::kFeatureGroupCount) {
        throw FormatError("invalid feature group count; expected 5");
    }

    const size_t groups = features.group_count;
    const size_t dimensions = features.feature_count;
    const size_t value_count = detail::checked_product(
        {static_cast<size_t>(features.frame_count), dimensions},
        "feature value dimensions");
    features.group_starts = detail::read_guarded_vector<uint32_t>(
        input, input_size, groups, "feature group starts");
    features.group_stops = detail::read_guarded_vector<uint32_t>(
        input, input_size, groups, "feature group stops");
    features.offsets = detail::read_guarded_vector<float>(
        input, input_size, dimensions, "feature offsets");
    features.scales = detail::read_guarded_vector<float>(
        input, input_size, dimensions, "feature scales");
    features.values = detail::read_guarded_vector<float>(
        input, input_size, value_count, "feature values");

    detail::require_eof(input, "feature file");
    detail::validate_features(features);
    return features;
}

}  // namespace interaction
