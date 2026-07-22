#include "g1_skeleton.h"
#include "interaction_database.h"
#include "interaction_trajectory_database.h"

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
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

struct BinaryFixture {
    std::vector<uint8_t> bytes;
    std::map<std::string, size_t> offsets;
};

void append_u8(std::vector<uint8_t>& output, uint8_t value) {
    output.push_back(value);
}

void append_u32(std::vector<uint8_t>& output, uint32_t value) {
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
        output.push_back(static_cast<uint8_t>(value >> (byte * 8U)));
    }
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

void mark(BinaryFixture& fixture, std::string name) {
    fixture.offsets.emplace(std::move(name), fixture.bytes.size());
}

void append_floats(
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
    fixture.bytes.insert(fixture.bytes.end(), {'G', '1', 'I', 'N', 'T', 'D', 'B', '1'});
    for (uint32_t value : std::array<uint32_t, 8>{
             1, 0x01020304U, 25, 1, kFrameCount, kBoneCount, kClipCount,
             kHandDofCount}) {
        append_u32(fixture.bytes, value);
    }

    mark(fixture, "parents");
    for (int32_t parent : g1_skeleton::kParents) append_i32(fixture.bytes, parent);
    mark(fixture, "range_starts");
    append_i32(fixture.bytes, 0);
    mark(fixture, "range_stops");
    append_i32(fixture.bytes, static_cast<int32_t>(kFrameCount));

    const size_t frame_bones = kFrameCount * kBoneCount;
    append_floats(fixture, "positions", frame_bones * 3U, 0.25F);
    append_floats(fixture, "velocities", frame_bones * 3U, 1000.25F);
    append_identity_quaternions(fixture, "rotations", frame_bones);
    append_floats(fixture, "angular_velocities", frame_bones * 3U, 2000.25F);
    mark(fixture, "foot_contacts");
    for (size_t frame = 0; frame < kFrameCount; ++frame) {
        append_u8(fixture.bytes, static_cast<uint8_t>(frame % 2U));
        append_u8(fixture.bytes, static_cast<uint8_t>((frame + 1U) % 2U));
    }
    mark(fixture, "hand_contacts");
    for (size_t frame = 0; frame < kFrameCount; ++frame) {
        append_u8(fixture.bytes, 0);
        append_u8(fixture.bytes, static_cast<uint8_t>(frame >= 2U));
    }
    append_floats(fixture, "hand_dof", kFrameCount * kHandDofCount, 3000.25F);
    append_floats(
        fixture, "hand_dof_velocities", kFrameCount * kHandDofCount, 4000.25F);
    mark(fixture, "phases");
    for (uint8_t phase : std::array<uint8_t, kFrameCount>{0, 1, 2, 3, 4, 4, 4, 4, 4}) {
        append_u8(fixture.bytes, phase);
    }
    mark(fixture, "active_hands");
    append_u8(fixture.bytes, 1);
    mark(fixture, "time_to_contact");
    for (float value : std::array<float, kFrameCount>{2, 1, 0, 0, 0, 0, 0, 0, 0}) {
        append_f32(fixture.bytes, value);
    }
    append_floats(fixture, "object_positions", kFrameCount * 3U, 5000.25F);
    append_identity_quaternions(fixture, "object_rotations", kFrameCount);
    append_floats(fixture, "object_velocities", kFrameCount * 3U, 6000.25F);
    append_floats(
        fixture, "object_angular_velocities", kFrameCount * 3U, 7000.25F);
    append_floats(fixture, "table_positions", kClipCount * 3U, 8000.25F);
    append_identity_quaternions(fixture, "table_rotations", kClipCount);
    mark(fixture, "table_sizes");
    for (float value : std::array<float, 3>{1, 2, 3}) append_f32(fixture.bytes, value);
    mark(fixture, "object_dimensions");
    for (float value : std::array<float, 3>{0.1F, 0.2F, 0.3F}) {
        append_f32(fixture.bytes, value);
    }
    append_floats(fixture, "grasp_positions_object", kClipCount * 3U, 9000.25F);
    append_identity_quaternions(fixture, "grasp_rotations_object", kClipCount);
    mark(fixture, "approach_directions_object");
    append_f32(fixture.bytes, 1);
    append_f32(fixture.bytes, 0);
    append_f32(fixture.bytes, 0);
    mark(fixture, "source_frames");
    for (uint32_t frame = 0; frame < kFrameCount; ++frame) {
        append_i32(fixture.bytes, static_cast<int32_t>(frame));
    }
    return fixture;
}

void overwrite_u32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    assert(offset + sizeof(value) <= bytes.size());
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
        bytes[offset + byte] = static_cast<uint8_t>(value >> (byte * 8U));
    }
}

void overwrite_i32(std::vector<uint8_t>& bytes, size_t offset, int32_t value) {
    overwrite_u32(bytes, offset, static_cast<uint32_t>(value));
}

void overwrite_f32(std::vector<uint8_t>& bytes, size_t offset, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    overwrite_u32(bytes, offset, bits);
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto suffix = std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count();
        path_ = std::filesystem::temp_directory_path() /
                ("interaction-trajectory-database-test-" + std::to_string(suffix));
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

void write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("could not create test fixture");
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("could not write test fixture");
}

template<class Loader>
void expect_format_error(
    const std::filesystem::path& path,
    const std::vector<uint8_t>& bytes,
    std::string_view label,
    Loader loader) {
    write_bytes(path, bytes);
    bool threw = false;
    try {
        loader(path);
    } catch (const interaction::FormatError& error) {
        threw = true;
        assert(std::string(error.what()).find(label) != std::string::npos);
    }
    assert(threw);
}

void test_compact_loader_matches_retained_full_database(const TemporaryDirectory& temporary) {
    const BinaryFixture fixture = make_database_fixture();
    const auto path = temporary.path() / "valid.bin";
    write_bytes(path, fixture.bytes);

    const interaction::Database full = interaction::load_database(path);
    const interaction::Database compact = interaction::load_trajectory_database(path);
    assert(full.parents == compact.parents);
    assert(full.range_starts == compact.range_starts);
    assert(full.range_stops == compact.range_stops);
    assert(full.positions == compact.positions);
    assert(full.rotations == compact.rotations);
    assert(full.phases == compact.phases);
    assert(full.active_hands == compact.active_hands);
    assert(full.object_positions == compact.object_positions);
    assert(full.object_rotations == compact.object_rotations);
    assert(full.table_positions == compact.table_positions);
    assert(full.table_rotations == compact.table_rotations);
    assert(full.table_sizes == compact.table_sizes);
    assert(full.object_dimensions == compact.object_dimensions);
    assert(full.grasp_positions_object == compact.grasp_positions_object);
    assert(full.grasp_rotations_object == compact.grasp_rotations_object);
    assert(full.approach_directions_object == compact.approach_directions_object);
    assert(compact.velocities.empty());
    assert(compact.angular_velocities.empty());
    assert(compact.foot_contacts.empty());
    assert(compact.hand_contacts.empty());
    assert(compact.hand_dof.empty());
    assert(compact.hand_dof_velocities.empty());
    assert(compact.time_to_contact.empty());
    assert(compact.object_velocities.empty());
    assert(compact.object_angular_velocities.empty());
    assert(compact.source_frames.empty());
}

void test_compact_loader_rejects_malformed_input(const TemporaryDirectory& temporary) {
    const BinaryFixture fixture = make_database_fixture();
    const auto path = temporary.path() / "invalid.bin";
    const auto load = [](const std::filesystem::path& value) {
        static_cast<void>(interaction::load_trajectory_database(value));
    };

    auto bytes = fixture.bytes;
    bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(
        fixture.offsets.at("velocities") + sizeof(float)));
    expect_format_error(path, bytes, "schema-v1 byte count", load);

    bytes = fixture.bytes;
    bytes.push_back(0xFFU);
    expect_format_error(path, bytes, "schema-v1 byte count", load);

    bytes = fixture.bytes;
    overwrite_i32(bytes, fixture.offsets.at("range_stops"), 3);
    expect_format_error(path, bytes, "range_stops", load);

    bytes = fixture.bytes;
    bytes[fixture.offsets.at("phases") + 3U] = 1;
    expect_format_error(path, bytes, "phases", load);

    bytes = fixture.bytes;
    bytes[fixture.offsets.at("phases") + 2U] = 4;
    bytes[fixture.offsets.at("phases") + 3U] = 4;
    expect_format_error(path, bytes, "phases", load);

    bytes = fixture.bytes;
    overwrite_f32(bytes, fixture.offsets.at("rotations"), 2.0F);
    expect_format_error(path, bytes, "rotations", load);

    bytes = fixture.bytes;
    bytes[fixture.offsets.at("active_hands")] = 2;
    expect_format_error(path, bytes, "active_hands", load);

    bytes = fixture.bytes;
    overwrite_f32(
        bytes,
        fixture.offsets.at("approach_directions_object") + sizeof(float),
        0.5F);
    expect_format_error(path, bytes, "must be horizontal", load);

    bytes = fixture.bytes;
    overwrite_f32(
        bytes, fixture.offsets.at("approach_directions_object"), 0.5F);
    expect_format_error(path, bytes, "must contain unit directions", load);
}

}  // namespace

int main() {
    const TemporaryDirectory temporary;
    test_compact_loader_matches_retained_full_database(temporary);
    test_compact_loader_rejects_malformed_input(temporary);
}
