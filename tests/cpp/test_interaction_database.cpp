#include "g1_skeleton.h"
#include "interaction_database.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kFrameCount = 9;
constexpr uint32_t kBoneCount = 31;
constexpr uint32_t kClipCount = 1;
constexpr uint32_t kHandDofCount = 14;
constexpr uint32_t kFeatureCount = 71;
constexpr uint32_t kFeatureGroupCount = 5;

struct BinaryFixture {
    std::vector<uint8_t> bytes;
    std::map<std::string, size_t> offsets;
};

void append_u8(std::vector<uint8_t>& output, uint8_t value) {
    output.push_back(value);
}

void append_u32(std::vector<uint8_t>& output, uint32_t value) {
    output.push_back(static_cast<uint8_t>(value));
    output.push_back(static_cast<uint8_t>(value >> 8U));
    output.push_back(static_cast<uint8_t>(value >> 16U));
    output.push_back(static_cast<uint8_t>(value >> 24U));
}

void append_i32(std::vector<uint8_t>& output, int32_t value) {
    append_u32(output, static_cast<uint32_t>(value));
}

void append_f32(std::vector<uint8_t>& output, float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(output, bits);
}

void append_magic(std::vector<uint8_t>& output, std::string_view magic) {
    output.insert(output.end(), magic.begin(), magic.end());
}

void overwrite_u32(std::vector<uint8_t>& output, size_t offset, uint32_t value) {
    assert(offset + sizeof(value) <= output.size());
    output[offset] = static_cast<uint8_t>(value);
    output[offset + 1] = static_cast<uint8_t>(value >> 8U);
    output[offset + 2] = static_cast<uint8_t>(value >> 16U);
    output[offset + 3] = static_cast<uint8_t>(value >> 24U);
}

void overwrite_i32(std::vector<uint8_t>& output, size_t offset, int32_t value) {
    overwrite_u32(output, offset, static_cast<uint32_t>(value));
}

void overwrite_f32(std::vector<uint8_t>& output, size_t offset, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    overwrite_u32(output, offset, bits);
}

void mark(BinaryFixture& fixture, std::string name) {
    fixture.offsets.emplace(std::move(name), fixture.bytes.size());
}

void append_float_series(
    BinaryFixture& fixture,
    std::string name,
    size_t count,
    float first) {
    mark(fixture, std::move(name));
    for (size_t index = 0; index < count; ++index) {
        append_f32(fixture.bytes, first + static_cast<float>(index));
    }
}

void append_identity_quaternions(
    BinaryFixture& fixture,
    std::string name,
    size_t count) {
    mark(fixture, std::move(name));
    for (size_t index = 0; index < count; ++index) {
        append_f32(fixture.bytes, 1.0F);
        append_f32(fixture.bytes, 0.0F);
        append_f32(fixture.bytes, 0.0F);
        append_f32(fixture.bytes, 0.0F);
    }
}

BinaryFixture make_database_fixture() {
    BinaryFixture fixture;
    append_magic(fixture.bytes, "G1INTDB1");
    append_u32(fixture.bytes, 1);
    append_u32(fixture.bytes, 0x01020304U);
    append_u32(fixture.bytes, 25);
    append_u32(fixture.bytes, 1);
    append_u32(fixture.bytes, kFrameCount);
    append_u32(fixture.bytes, kBoneCount);
    append_u32(fixture.bytes, kClipCount);
    append_u32(fixture.bytes, kHandDofCount);

    mark(fixture, "parents");
    for (int32_t parent : g1_skeleton::kParents) {
        append_i32(fixture.bytes, parent);
    }

    mark(fixture, "range_starts");
    append_i32(fixture.bytes, 0);
    mark(fixture, "range_stops");
    append_i32(fixture.bytes, static_cast<int32_t>(kFrameCount));

    const size_t bone_vector_count = kFrameCount * kBoneCount * 3U;
    append_float_series(fixture, "positions", bone_vector_count, 0.25F);
    append_float_series(fixture, "velocities", bone_vector_count, 1000.25F);
    append_identity_quaternions(
        fixture, "rotations", kFrameCount * kBoneCount);
    append_float_series(
        fixture, "angular_velocities", bone_vector_count, 2000.25F);

    mark(fixture, "foot_contacts");
    for (uint32_t frame = 0; frame < kFrameCount; ++frame) {
        append_u8(fixture.bytes, static_cast<uint8_t>(frame % 2U));
        append_u8(fixture.bytes, static_cast<uint8_t>((frame + 1U) % 2U));
    }

    mark(fixture, "hand_contacts");
    for (uint32_t frame = 0; frame < kFrameCount; ++frame) {
        append_u8(fixture.bytes, 0);
        append_u8(fixture.bytes, static_cast<uint8_t>(frame >= 2U));
    }

    append_float_series(
        fixture, "hand_dof", kFrameCount * kHandDofCount, 3000.25F);
    append_float_series(
        fixture,
        "hand_dof_velocities",
        kFrameCount * kHandDofCount,
        4000.25F);

    mark(fixture, "phases");
    for (uint8_t phase : std::vector<uint8_t>{0, 1, 2, 3, 4, 4, 4, 4, 4}) {
        append_u8(fixture.bytes, phase);
    }

    mark(fixture, "active_hands");
    append_u8(fixture.bytes, 1);

    mark(fixture, "time_to_contact");
    for (float value : std::vector<float>{2.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                                          0.0F, 0.0F, 0.0F, 0.0F}) {
        append_f32(fixture.bytes, value);
    }

    append_float_series(
        fixture, "object_positions", kFrameCount * 3U, 5000.25F);
    append_identity_quaternions(fixture, "object_rotations", kFrameCount);
    append_float_series(
        fixture, "object_velocities", kFrameCount * 3U, 6000.25F);
    append_float_series(
        fixture,
        "object_angular_velocities",
        kFrameCount * 3U,
        7000.25F);

    append_float_series(fixture, "table_positions", kClipCount * 3U, 8000.25F);
    append_identity_quaternions(fixture, "table_rotations", kClipCount);

    mark(fixture, "table_sizes");
    for (float value : std::vector<float>{1.0F, 2.0F, 3.0F}) {
        append_f32(fixture.bytes, value);
    }
    mark(fixture, "object_dimensions");
    for (float value : std::vector<float>{0.1F, 0.2F, 0.3F}) {
        append_f32(fixture.bytes, value);
    }

    append_float_series(
        fixture, "grasp_positions_object", kClipCount * 3U, 9000.25F);
    append_identity_quaternions(
        fixture, "grasp_rotations_object", kClipCount);

    mark(fixture, "approach_directions_object");
    append_f32(fixture.bytes, 1.0F);
    append_f32(fixture.bytes, 0.0F);
    append_f32(fixture.bytes, 0.0F);

    mark(fixture, "source_frames");
    for (uint32_t frame = 0; frame < kFrameCount; ++frame) {
        append_i32(fixture.bytes, static_cast<int32_t>(frame + 10U));
    }
    return fixture;
}

BinaryFixture make_feature_fixture() {
    BinaryFixture fixture;
    append_magic(fixture.bytes, "G1INTFT1");
    append_u32(fixture.bytes, 1);
    append_u32(fixture.bytes, 0x01020304U);
    append_u32(fixture.bytes, kFrameCount);
    append_u32(fixture.bytes, kFeatureCount);
    append_u32(fixture.bytes, kFeatureGroupCount);

    mark(fixture, "group_starts");
    for (uint32_t value : std::vector<uint32_t>{0, 33, 45, 57, 65}) {
        append_u32(fixture.bytes, value);
    }
    mark(fixture, "group_stops");
    for (uint32_t value : std::vector<uint32_t>{33, 45, 57, 65, 71}) {
        append_u32(fixture.bytes, value);
    }

    append_float_series(fixture, "offsets", kFeatureCount, 0.5F);
    append_float_series(fixture, "scales", kFeatureCount, 1.0F);
    append_float_series(
        fixture, "values", kFrameCount * kFeatureCount, 100.5F);
    return fixture;
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto suffix = std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count();
        path_ = std::filesystem::temp_directory_path() /
                ("interaction-database-test-" + std::to_string(suffix));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

void write_bytes(
    const std::filesystem::path& path,
    const std::vector<uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not create test fixture");
    }
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("could not write test fixture");
    }
}

template<class Loader>
void expect_format_error(
    const std::filesystem::path& path,
    const std::vector<uint8_t>& bytes,
    std::string_view field,
    Loader loader) {
    write_bytes(path, bytes);
    bool threw = false;
    try {
        loader(path);
    } catch (const interaction::FormatError& error) {
        threw = true;
        assert(std::string(error.what()).find(field) != std::string::npos);
    }
    assert(threw);
}

void test_loads_database_and_features(const TemporaryDirectory& temporary) {
    const BinaryFixture database_fixture = make_database_fixture();
    const BinaryFixture feature_fixture = make_feature_fixture();
    const auto database_path = temporary.path() / "valid-database.bin";
    const auto feature_path = temporary.path() / "valid-features.bin";
    write_bytes(database_path, database_fixture.bytes);
    write_bytes(feature_path, feature_fixture.bytes);

    const interaction::Database database = interaction::load_database(database_path);
    assert(database.version == 1);
    assert(database.endian_marker == 0x01020304U);
    assert(database.fps_numerator == 25);
    assert(database.fps_denominator == 1);
    assert(database.frame_count == kFrameCount);
    assert(database.bone_count == kBoneCount);
    assert(database.clip_count == kClipCount);
    assert(database.hand_dof_count == kHandDofCount);
    assert(database.parents.front() == -1 && database.parents.back() == 29);
    assert(database.range_starts.front() == 0);
    assert(database.range_stops.back() == static_cast<int32_t>(kFrameCount));
    assert(database.positions.front() == 0.25F);
    assert(database.positions.back() == 836.25F);
    assert(database.velocities.front() == 1000.25F);
    assert(database.velocities.back() == 1836.25F);
    assert(database.rotations.front() == 1.0F);
    assert(database.rotations.back() == 0.0F);
    assert(database.angular_velocities.front() == 2000.25F);
    assert(database.angular_velocities.back() == 2836.25F);
    assert(database.foot_contacts.front() == 0 && database.foot_contacts.back() == 1);
    assert(database.hand_contacts.front() == 0 && database.hand_contacts.back() == 1);
    assert(database.hand_dof.front() == 3000.25F);
    assert(database.hand_dof.back() == 3125.25F);
    assert(database.hand_dof_velocities.front() == 4000.25F);
    assert(database.hand_dof_velocities.back() == 4125.25F);
    assert(database.phases.front() == 0 && database.phases.back() == 4);
    assert(database.active_hands.front() == 1);
    assert(database.time_to_contact.front() == 2.0F);
    assert(database.time_to_contact.back() == 0.0F);
    assert(database.object_positions.front() == 5000.25F);
    assert(database.object_positions.back() == 5026.25F);
    assert(database.object_rotations.front() == 1.0F);
    assert(database.object_rotations.back() == 0.0F);
    assert(database.object_velocities.front() == 6000.25F);
    assert(database.object_velocities.back() == 6026.25F);
    assert(database.object_angular_velocities.front() == 7000.25F);
    assert(database.object_angular_velocities.back() == 7026.25F);
    assert(database.table_positions.front() == 8000.25F);
    assert(database.table_positions.back() == 8002.25F);
    assert(database.table_rotations.front() == 1.0F);
    assert(database.table_rotations.back() == 0.0F);
    assert(database.table_sizes.front() == 1.0F && database.table_sizes.back() == 3.0F);
    assert(database.object_dimensions.front() == 0.1F);
    assert(database.object_dimensions.back() == 0.3F);
    assert(database.grasp_positions_object.front() == 9000.25F);
    assert(database.grasp_positions_object.back() == 9002.25F);
    assert(database.grasp_rotations_object.front() == 1.0F);
    assert(database.grasp_rotations_object.back() == 0.0F);
    assert(database.approach_directions_object.front() == 1.0F);
    assert(database.approach_directions_object.back() == 0.0F);
    assert(database.source_frames.front() == 10);
    assert(database.source_frames.back() == 18);

    const interaction::Features features = interaction::load_features(feature_path);
    assert(features.version == 1);
    assert(features.endian_marker == 0x01020304U);
    assert(features.frame_count == kFrameCount);
    assert(features.feature_count == kFeatureCount);
    assert(features.group_count == kFeatureGroupCount);
    assert(features.group_starts.front() == 0 && features.group_starts.back() == 65);
    assert(features.group_stops.front() == 33 && features.group_stops.back() == 71);
    assert(features.offsets.front() == 0.5F && features.offsets.back() == 70.5F);
    assert(features.scales.front() == 1.0F && features.scales.back() == 71.0F);
    assert(features.values.front() == 100.5F);
    assert(features.values.back() == 738.5F);
}

void test_required_database_corruptions(const TemporaryDirectory& temporary) {
    const BinaryFixture fixture = make_database_fixture();
    const auto path = temporary.path() / "invalid-database.bin";
    const auto load = [](const std::filesystem::path& value) {
        static_cast<void>(interaction::load_database(value));
    };

    auto bytes = fixture.bytes;
    bytes[0] = 'X';
    expect_format_error(path, bytes, "database magic", load);

    bytes = fixture.bytes;
    overwrite_u32(bytes, 12, 0);
    expect_format_error(path, bytes, "database endian marker", load);

    bytes = fixture.bytes;
    overwrite_u32(bytes, 28, 30);
    expect_format_error(path, bytes, "database bone count", load);

    bytes = fixture.bytes;
    overwrite_i32(bytes, fixture.offsets.at("range_starts"), 1);
    expect_format_error(path, bytes, "range_starts", load);

    bytes = fixture.bytes;
    overwrite_f32(bytes, fixture.offsets.at("rotations"), 2.0F);
    expect_format_error(path, bytes, "rotations", load);

    bytes = fixture.bytes;
    bytes.pop_back();
    expect_format_error(path, bytes, "source_frames", load);

    bytes = fixture.bytes;
    append_u8(bytes, 0xFFU);
    expect_format_error(path, bytes, "trailing bytes", load);
}

void test_database_semantic_corruptions(const TemporaryDirectory& temporary) {
    const BinaryFixture fixture = make_database_fixture();
    const auto path = temporary.path() / "semantic-database.bin";
    const auto load = [](const std::filesystem::path& value) {
        static_cast<void>(interaction::load_database(value));
    };

    auto bytes = fixture.bytes;
    overwrite_i32(bytes, fixture.offsets.at("parents") + sizeof(int32_t), -1);
    expect_format_error(path, bytes, "parents", load);

    bytes = fixture.bytes;
    overwrite_f32(
        bytes,
        fixture.offsets.at("positions"),
        std::numeric_limits<float>::quiet_NaN());
    expect_format_error(path, bytes, "positions", load);

    bytes = fixture.bytes;
    bytes[fixture.offsets.at("phases")] = 5;
    expect_format_error(path, bytes, "phases", load);

    bytes = fixture.bytes;
    bytes[fixture.offsets.at("active_hands")] = 2;
    expect_format_error(path, bytes, "active_hands", load);

    bytes = fixture.bytes;
    overwrite_f32(bytes, fixture.offsets.at("time_to_contact") + 4U, 3.0F);
    expect_format_error(path, bytes, "time_to_contact", load);

    bytes = fixture.bytes;
    overwrite_f32(bytes, fixture.offsets.at("table_sizes"), 0.0F);
    expect_format_error(path, bytes, "table_sizes", load);

    bytes = fixture.bytes;
    overwrite_f32(
        bytes,
        fixture.offsets.at("approach_directions_object") + sizeof(float),
        0.5F);
    expect_format_error(path, bytes, "approach_directions_object", load);

    bytes = fixture.bytes;
    overwrite_i32(bytes, fixture.offsets.at("source_frames") + 4U, 9);
    expect_format_error(path, bytes, "source_frames", load);

    bytes = fixture.bytes;
    bytes[fixture.offsets.at("hand_contacts") + 2U * 2U + 1U] = 0;
    expect_format_error(path, bytes, "hand_contacts", load);
}

void test_feature_corruptions(const TemporaryDirectory& temporary) {
    const BinaryFixture fixture = make_feature_fixture();
    const auto path = temporary.path() / "invalid-features.bin";
    const auto load = [](const std::filesystem::path& value) {
        static_cast<void>(interaction::load_features(value));
    };

    auto bytes = fixture.bytes;
    bytes[0] = 'X';
    expect_format_error(path, bytes, "feature magic", load);

    bytes = fixture.bytes;
    overwrite_u32(bytes, 12, 0);
    expect_format_error(path, bytes, "feature endian marker", load);

    bytes = fixture.bytes;
    overwrite_u32(bytes, 20, 70);
    expect_format_error(path, bytes, "feature count", load);

    bytes = fixture.bytes;
    overwrite_u32(bytes, fixture.offsets.at("group_starts"), 1);
    expect_format_error(path, bytes, "feature group ranges", load);

    bytes = fixture.bytes;
    overwrite_f32(
        bytes,
        fixture.offsets.at("offsets"),
        std::numeric_limits<float>::infinity());
    expect_format_error(path, bytes, "feature offsets", load);

    bytes = fixture.bytes;
    overwrite_f32(bytes, fixture.offsets.at("scales"), 0.0F);
    expect_format_error(path, bytes, "feature scales", load);

    bytes = fixture.bytes;
    bytes.pop_back();
    expect_format_error(path, bytes, "feature values", load);

    bytes = fixture.bytes;
    append_u8(bytes, 0xFFU);
    expect_format_error(path, bytes, "trailing bytes", load);
}

}  // namespace

int main() {
    const TemporaryDirectory temporary;
    test_loads_database_and_features(temporary);
    test_required_database_corruptions(temporary);
    test_database_semantic_corruptions(temporary);
    test_feature_corruptions(temporary);
}
