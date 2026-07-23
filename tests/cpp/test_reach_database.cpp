#include "reach_database.h"
#include "reach_motion.h"

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

template<class T>
void write_scalar(std::ofstream& output, T value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template<class T>
void write_vector(std::ofstream& output, const std::vector<T>& values) {
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
}

std::filesystem::path temporary_directory() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        ("g1-reach-loader-" + std::to_string(stamp));
    std::filesystem::create_directories(path);
    return path;
}

void write_database(
    const std::filesystem::path& path,
    const std::vector<int32_t>& contact_frames = {0, 1},
    bool trailing = false) {
    constexpr uint32_t frames = 2U;
    constexpr uint32_t bones = 31U;
    constexpr uint32_t clips = 2U;
    std::ofstream output(path, std::ios::binary);
    const std::array<char, 8> magic = {'G','1','R','C','H','D','2','\0'};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    for (uint32_t value : {
             2U, 0x01020304U, 25U, 1U, frames, bones, clips, 1U}) {
        write_scalar(output, value);
    }
    write_vector(output, std::vector<int32_t>(
        g1_skeleton::kParents.begin(), g1_skeleton::kParents.end()));
    write_vector(output, std::vector<int32_t>{0, 1});
    write_vector(output, std::vector<int32_t>{1, 2});
    write_vector(output, contact_frames);
    std::vector<float> positions(frames * bones * 3U, 0.0F);
    positions[0] = 1.0F;
    positions[1] = 2.0F;
    positions[2] = 3.0F;
    write_vector(output, positions);
    write_vector(output, std::vector<float>(frames * bones * 3U, 0.0F));
    std::vector<float> rotations(frames * bones * 4U, 0.0F);
    for (size_t value = 0U; value < frames * bones; ++value) {
        rotations[value * 4U] = 1.0F;
    }
    write_vector(output, rotations);
    write_vector(output, std::vector<float>(frames * bones * 3U, 0.0F));
    write_vector(output, std::vector<uint8_t>{1, 0, 0, 1});
    write_vector(output, std::vector<uint8_t>{0, 1});
    write_vector(output, std::vector<uint8_t>{0, 1});
    write_vector(output, std::vector<uint32_t>{0, 0});
    write_vector(output, std::vector<int32_t>{-1, 0});
    write_vector(output, std::vector<float>{
        0.4F, 0.8F, 0.2F, 0.4F, 0.8F, -0.2F});
    write_vector(output, std::vector<float>{
        1,0,0,0, 1,0,0,0});
    write_vector(output, std::vector<float>{
        1,0,0, 1,0,0});
    write_vector(output, std::vector<int32_t>{100, 100});
    const std::string name = "pickup_north_0";
    write_scalar(output, static_cast<uint32_t>(name.size()));
    output.write(name.data(), static_cast<std::streamsize>(name.size()));
    if (trailing) output.put('x');
}

void write_features(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary);
    const std::array<char, 8> magic = {'G','1','R','C','H','F','2','\0'};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    for (uint32_t value : {2U, 0x01020304U, 2U, 10U}) {
        write_scalar(output, value);
    }
    write_vector(output, std::vector<float>{
        0.4F,0.8F,0.2F, 1,0,0, 1,0,0,0,
        0.4F,0.8F,-0.2F, 1,0,0, 1,0,0,0});
}

void test_loads_pack_and_reconstructs_pose() {
    const auto directory = temporary_directory();
    write_database(directory / "reach_database.bin");
    write_features(directory / "reach_features.bin");

    const reach::Pack pack = reach::load_pack(directory);

    assert(pack.database.frame_count == 2U);
    assert(pack.database.clip_count == 2U);
    assert(pack.database.source_names.size() == 1U);
    assert(pack.database.source_names[0] == "pickup_north_0");
    assert(pack.database.active_hands == std::vector<uint8_t>({0U, 1U}));
    assert(pack.database.augmentations == std::vector<uint8_t>({0U, 1U}));
    assert(pack.database.original_indices == std::vector<int32_t>({-1, 0}));
    assert(pack.database.contact_frames == std::vector<int32_t>({0, 1}));
    assert(pack.features.values.size() == 20U);
    const interaction::Pose pose = reach::pose_at_frame(pack.database, 0);
    assert(pose.positions[g1_skeleton::Simulation].x == 1.0F);
    assert(pose.positions[g1_skeleton::Simulation].y == 2.0F);
    assert(pose.positions[g1_skeleton::Simulation].z == 3.0F);
    assert(pose.rotations[g1_skeleton::Simulation].w == 1.0F);
    assert(pose.foot_contacts[0] == 1U);
    assert(pose.foot_contacts[1] == 0U);
    const interaction::Transform endpoint = reach::endpoint_transform(
        pack.database, 1U);
    assert(endpoint.position.z == -0.2F);
    assert(reach::clip_contact_frame(pack.database, 0U) == 0);
    assert(reach::clip_return_start(pack.database, 0U) == 1);
    assert(reach::clip_return_stop(pack.database, 0U) == 1);
    assert(reach::clip_contact_frame(pack.database, 1U) == 1);
    assert(reach::clip_return_start(pack.database, 1U) == 2);
    assert(reach::clip_return_stop(pack.database, 1U) == 2);
    std::filesystem::remove_all(directory);
}

void test_rejects_trailing_database_bytes() {
    const auto directory = temporary_directory();
    write_database(directory / "reach_database.bin", {0, 1}, true);
    write_features(directory / "reach_features.bin");
    bool threw = false;
    try {
        (void)reach::load_pack(directory);
    } catch (const interaction::FormatError&) {
        threw = true;
    }
    assert(threw);
    std::filesystem::remove_all(directory);
}

void test_rejects_contact_before_clip_start() {
    const auto directory = temporary_directory();
    write_database(directory / "reach_database.bin", {-1, 1});
    write_features(directory / "reach_features.bin");
    bool threw = false;
    try {
        (void)reach::load_pack(directory);
    } catch (const interaction::FormatError&) {
        threw = true;
    }
    assert(threw);
    std::filesystem::remove_all(directory);
}

void test_rejects_contact_at_clip_stop() {
    const auto directory = temporary_directory();
    write_database(directory / "reach_database.bin", {1, 2});
    write_features(directory / "reach_features.bin");
    bool threw = false;
    try {
        (void)reach::load_pack(directory);
    } catch (const interaction::FormatError&) {
        threw = true;
    }
    assert(threw);
    std::filesystem::remove_all(directory);
}

}  // namespace

int main(int argc, char** argv) {
    test_loads_pack_and_reconstructs_pose();
    test_rejects_trailing_database_bytes();
    test_rejects_contact_before_clip_start();
    test_rejects_contact_at_clip_stop();
    if (argc == 2) {
        const reach::Pack pack = reach::load_pack(argv[1]);
        assert(pack.database.clip_count == 2U);
        assert(pack.database.active_hands == std::vector<uint8_t>({0U, 1U}));
        assert(pack.database.augmentations == std::vector<uint8_t>({0U, 1U}));
        assert(pack.database.original_indices == std::vector<int32_t>({-1, 0}));
    } else {
        assert(argc == 1);
    }
}
