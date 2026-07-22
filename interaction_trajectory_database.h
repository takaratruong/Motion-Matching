#pragma once

#include "interaction_database.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

namespace interaction {

template<class T>
void skip_vector(
    std::istream& input,
    size_t input_size,
    size_t count,
    std::string_view label) {
    static_assert(std::is_trivially_copyable_v<T>);
    const size_t byte_count = detail::checked_multiply(count, sizeof(T), label);
    if (byte_count >
        static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        throw FormatError("overflow " + std::string(label));
    }
    const std::streampos position = input.tellg();
    if (position < 0) {
        throw FormatError("cannot determine offset for " + std::string(label));
    }
    const auto offset = static_cast<uintmax_t>(position);
    if (offset > input_size || byte_count > input_size - offset) {
        throw FormatError("truncated " + std::string(label));
    }
    input.seekg(static_cast<std::streamoff>(byte_count), std::ios::cur);
    if (!input) {
        throw FormatError("truncated " + std::string(label));
    }
    const std::streampos end = input.tellg();
    if (end < 0 || static_cast<uintmax_t>(end) != offset + byte_count) {
        throw FormatError("could not skip " + std::string(label));
    }
}

namespace trajectory_detail {

inline size_t checked_add(
    size_t left,
    size_t right,
    std::string_view label) {
    if (left > std::numeric_limits<size_t>::max() - right) {
        throw FormatError("overflow " + std::string(label));
    }
    return left + right;
}

inline size_t add_vector_bytes(
    size_t total,
    size_t count,
    size_t element_size,
    std::string_view label) {
    return checked_add(
        total,
        detail::checked_multiply(count, element_size, label),
        "schema-v1 byte count");
}

inline size_t schema_v1_byte_count(
    size_t frames,
    size_t bones,
    size_t clips,
    size_t hand_dofs) {
    size_t total = 0;
    total = checked_add(total, 8U, "schema-v1 byte count");
    total = checked_add(
        total,
        detail::checked_multiply(8U, sizeof(uint32_t), "schema-v1 header"),
        "schema-v1 byte count");

    const size_t frame_bones =
        detail::checked_product({frames, bones}, "frame/bone dimensions");
    const size_t bone_vectors =
        detail::checked_product({frame_bones, 3U}, "bone vector dimensions");
    const size_t bone_quaternions = detail::checked_product(
        {frame_bones, 4U}, "bone quaternion dimensions");
    const size_t frame_pairs =
        detail::checked_product({frames, 2U}, "contact dimensions");
    const size_t hand_values =
        detail::checked_product({frames, hand_dofs}, "hand dof dimensions");
    const size_t frame_vectors =
        detail::checked_product({frames, 3U}, "frame vector dimensions");
    const size_t frame_quaternions =
        detail::checked_product({frames, 4U}, "frame quaternion dimensions");
    const size_t clip_vectors =
        detail::checked_product({clips, 3U}, "clip vector dimensions");
    const size_t clip_quaternions =
        detail::checked_product({clips, 4U}, "clip quaternion dimensions");

    total = add_vector_bytes(total, bones, sizeof(int32_t), "parents");
    total = add_vector_bytes(total, clips, sizeof(int32_t), "range_starts");
    total = add_vector_bytes(total, clips, sizeof(int32_t), "range_stops");
    total = add_vector_bytes(total, bone_vectors, sizeof(float), "positions");
    total = add_vector_bytes(total, bone_vectors, sizeof(float), "velocities");
    total = add_vector_bytes(total, bone_quaternions, sizeof(float), "rotations");
    total = add_vector_bytes(
        total, bone_vectors, sizeof(float), "angular_velocities");
    total = add_vector_bytes(total, frame_pairs, sizeof(uint8_t), "foot_contacts");
    total = add_vector_bytes(total, frame_pairs, sizeof(uint8_t), "hand_contacts");
    total = add_vector_bytes(total, hand_values, sizeof(float), "hand_dof");
    total = add_vector_bytes(
        total, hand_values, sizeof(float), "hand_dof_velocities");
    total = add_vector_bytes(total, frames, sizeof(uint8_t), "phases");
    total = add_vector_bytes(total, clips, sizeof(uint8_t), "active_hands");
    total = add_vector_bytes(total, frames, sizeof(float), "time_to_contact");
    total = add_vector_bytes(
        total, frame_vectors, sizeof(float), "object_positions");
    total = add_vector_bytes(
        total, frame_quaternions, sizeof(float), "object_rotations");
    total = add_vector_bytes(
        total, frame_vectors, sizeof(float), "object_velocities");
    total = add_vector_bytes(
        total, frame_vectors, sizeof(float), "object_angular_velocities");
    total = add_vector_bytes(
        total, clip_vectors, sizeof(float), "table_positions");
    total = add_vector_bytes(
        total, clip_quaternions, sizeof(float), "table_rotations");
    total = add_vector_bytes(total, clip_vectors, sizeof(float), "table_sizes");
    total = add_vector_bytes(
        total, clip_vectors, sizeof(float), "object_dimensions");
    total = add_vector_bytes(
        total, clip_vectors, sizeof(float), "grasp_positions_object");
    total = add_vector_bytes(
        total, clip_quaternions, sizeof(float), "grasp_rotations_object");
    total = add_vector_bytes(
        total, clip_vectors, sizeof(float), "approach_directions_object");
    total = add_vector_bytes(total, frames, sizeof(int32_t), "source_frames");
    return total;
}

inline void validate_trajectory_phases(const Database& database) {
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
        uint8_t previous = database.phases[start];
        for (size_t frame = start; frame < stop; ++frame) {
            const uint8_t phase = database.phases[frame];
            if (frame != start && phase < previous) {
                throw FormatError("phases must be monotonic within each range");
            }
            previous = phase;
            has_contact = has_contact || phase == 2U;
            has_lift = has_lift || phase == 3U;
            if (phase == 4U) ++hold_count;
        }
        if (!has_contact || !has_lift || hold_count < detail::kHoldContactSamples) {
            throw FormatError(
                "phases in each range must include CONTACT, LIFT, and at least 5 HOLD samples");
        }

        const size_t approach = clip * 3U;
        const double x = database.approach_directions_object[approach];
        const double y = database.approach_directions_object[approach + 1U];
        const double z = database.approach_directions_object[approach + 2U];
        if (std::abs(y) > detail::kFloatTolerance) {
            throw FormatError(
                "approach_directions_object must be horizontal");
        }
        const double norm = std::sqrt(x * x + y * y + z * z);
        if (std::abs(norm - 1.0) > detail::kFloatTolerance) {
            throw FormatError(
                "approach_directions_object must contain unit directions");
        }
    }
}

inline void validate_trajectory_database(const Database& database) {
    if (!std::equal(
            database.parents.begin(),
            database.parents.end(),
            g1_skeleton::kParents.begin())) {
        throw FormatError("parents must match the exact G1 parent array");
    }
    detail::validate_ranges(database);
    detail::require_finite(database.positions, "positions");
    detail::require_finite(database.rotations, "rotations");
    detail::require_finite(database.object_positions, "object_positions");
    detail::require_finite(database.object_rotations, "object_rotations");
    detail::require_finite(database.table_positions, "table_positions");
    detail::require_finite(database.table_rotations, "table_rotations");
    detail::require_finite(database.table_sizes, "table_sizes");
    detail::require_finite(database.object_dimensions, "object_dimensions");
    detail::require_finite(
        database.grasp_positions_object, "grasp_positions_object");
    detail::require_finite(
        database.grasp_rotations_object, "grasp_rotations_object");
    detail::require_finite(
        database.approach_directions_object, "approach_directions_object");
    detail::require_binary(database.active_hands, "active_hands");
    detail::require_unit_quaternions(database.rotations, "rotations");
    detail::require_unit_quaternions(database.object_rotations, "object_rotations");
    detail::require_unit_quaternions(database.table_rotations, "table_rotations");
    detail::require_unit_quaternions(
        database.grasp_rotations_object, "grasp_rotations_object");
    if (!std::all_of(
            database.table_sizes.begin(), database.table_sizes.end(),
            [](float value) { return value > 0.0F; })) {
        throw FormatError("table_sizes must be positive");
    }
    if (!std::all_of(
            database.object_dimensions.begin(), database.object_dimensions.end(),
            [](float value) { return value > 0.0F; })) {
        throw FormatError("object_dimensions must be positive");
    }
    validate_trajectory_phases(database);
}

}  // namespace trajectory_detail

inline Database load_trajectory_database(const std::filesystem::path& path) {
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
    const size_t bone_vectors =
        detail::checked_product({frame_bones, 3U}, "bone vector dimensions");
    const size_t bone_quaternions = detail::checked_product(
        {frame_bones, 4U}, "bone quaternion dimensions");
    const size_t frame_pairs =
        detail::checked_product({frames, 2U}, "contact dimensions");
    const size_t hand_values =
        detail::checked_product({frames, hand_dofs}, "hand dof dimensions");
    const size_t frame_vectors =
        detail::checked_product({frames, 3U}, "frame vector dimensions");
    const size_t frame_quaternions =
        detail::checked_product({frames, 4U}, "frame quaternion dimensions");
    const size_t clip_vectors =
        detail::checked_product({clips, 3U}, "clip vector dimensions");
    const size_t clip_quaternions =
        detail::checked_product({clips, 4U}, "clip quaternion dimensions");

    const size_t expected_size = trajectory_detail::schema_v1_byte_count(
        frames, bones, clips, hand_dofs);
    if (input_size != expected_size) {
        throw FormatError("schema-v1 byte count mismatch");
    }

    database.parents = detail::read_guarded_vector<int32_t>(
        input, input_size, bones, "parents");
    database.range_starts = detail::read_guarded_vector<int32_t>(
        input, input_size, clips, "range_starts");
    database.range_stops = detail::read_guarded_vector<int32_t>(
        input, input_size, clips, "range_stops");
    database.positions = detail::read_guarded_vector<float>(
        input, input_size, bone_vectors, "positions");
    skip_vector<float>(input, input_size, bone_vectors, "velocities");
    database.rotations = detail::read_guarded_vector<float>(
        input, input_size, bone_quaternions, "rotations");
    skip_vector<float>(input, input_size, bone_vectors, "angular_velocities");
    skip_vector<uint8_t>(input, input_size, frame_pairs, "foot_contacts");
    skip_vector<uint8_t>(input, input_size, frame_pairs, "hand_contacts");
    skip_vector<float>(input, input_size, hand_values, "hand_dof");
    skip_vector<float>(input, input_size, hand_values, "hand_dof_velocities");
    database.phases = detail::read_guarded_vector<uint8_t>(
        input, input_size, frames, "phases");
    database.active_hands = detail::read_guarded_vector<uint8_t>(
        input, input_size, clips, "active_hands");
    skip_vector<float>(input, input_size, frames, "time_to_contact");
    database.object_positions = detail::read_guarded_vector<float>(
        input, input_size, frame_vectors, "object_positions");
    database.object_rotations = detail::read_guarded_vector<float>(
        input, input_size, frame_quaternions, "object_rotations");
    skip_vector<float>(input, input_size, frame_vectors, "object_velocities");
    skip_vector<float>(
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
    skip_vector<int32_t>(input, input_size, frames, "source_frames");

    const std::streampos end = input.tellg();
    if (end < 0 || static_cast<uintmax_t>(end) != input_size) {
        throw FormatError("database did not end at expected EOF");
    }
    detail::require_eof(input, "database");
    trajectory_detail::validate_trajectory_database(database);
    return database;
}

}  // namespace interaction
