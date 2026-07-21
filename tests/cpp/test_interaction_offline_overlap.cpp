#include "interaction_offline_overlap.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::array<char, 8> kMagic = {
    'G', '1', 'O', 'V', 'L', 'P', '0', '1'};
constexpr const char* kSignature =
    "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7";

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void write_u32(std::ofstream& stream, uint32_t value) {
    const std::array<uint8_t, 4> bytes = {
        static_cast<uint8_t>(value & 0xFFU),
        static_cast<uint8_t>((value >> 8U) & 0xFFU),
        static_cast<uint8_t>((value >> 16U) & 0xFFU),
        static_cast<uint8_t>((value >> 24U) & 0xFFU),
    };
    stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void write_f32(std::ofstream& stream, float value) {
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be IEEE-754 binary32");
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(stream, bits);
}

struct FixtureOptions {
    uint32_t frame_count = 80U;
    uint32_t bone_count = 31U;
    uint32_t contact_count = 2U;
    std::string signature = kSignature;
    bool nonfinite_position = false;
    bool nonunit_quaternion = false;
    bool trailing_byte = false;
};

std::filesystem::path fixture_path(const char* name) {
    const std::filesystem::path directory = "build/tests";
    std::filesystem::create_directories(directory);
    return directory / name;
}

void write_fixture(const std::filesystem::path& path, FixtureOptions options = {}) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(stream.is_open(), "could not create offline-overlap fixture");
    stream.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_u32(stream, 1U);
    write_u32(stream, 25U);
    write_u32(stream, options.frame_count);
    write_u32(stream, options.bone_count);
    write_u32(stream, options.contact_count);
    write_u32(stream, static_cast<uint32_t>(options.signature.size()));
    stream.write(options.signature.data(),
                 static_cast<std::streamsize>(options.signature.size()));
    for (const std::string_view name : g1_skeleton::kBoneNames) {
        write_u32(stream, static_cast<uint32_t>(name.size()));
        stream.write(name.data(), static_cast<std::streamsize>(name.size()));
    }
    for (uint32_t frame = 0U; frame < options.frame_count; ++frame) {
        for (uint32_t bone = 0U; bone < options.bone_count; ++bone) {
            for (uint32_t component = 0U; component < 3U; ++component) {
                float value = static_cast<float>(frame * 100U + bone * 3U + component);
                if (options.nonfinite_position && frame == 0U && bone == 0U && component == 0U) {
                    value = std::numeric_limits<float>::quiet_NaN();
                }
                write_f32(stream, value);
            }
        }
    }
    for (uint32_t field = 0U; field < 2U; ++field) {
        for (uint32_t frame = 0U; frame < options.frame_count; ++frame) {
            for (uint32_t bone = 0U; bone < options.bone_count; ++bone) {
                for (uint32_t component = 0U; component < 3U; ++component) {
                    write_f32(stream, static_cast<float>(field + component));
                }
            }
        }
    }
    for (uint32_t frame = 0U; frame < options.frame_count; ++frame) {
        for (uint32_t bone = 0U; bone < options.bone_count; ++bone) {
            write_f32(stream, options.nonunit_quaternion && frame == 0U && bone == 0U ? 0.5F : 1.0F);
            write_f32(stream, 0.0F);
            write_f32(stream, 0.0F);
            write_f32(stream, 0.0F);
        }
    }
    for (uint32_t frame = 0U; frame < options.frame_count; ++frame) {
        for (uint32_t contact = 0U; contact < options.contact_count; ++contact) {
            const uint8_t value = static_cast<uint8_t>((frame + contact) % 2U);
            stream.write(reinterpret_cast<const char*>(&value), 1);
        }
    }
    if (options.trailing_byte) stream.put('x');
    require(stream.good(), "could not write offline-overlap fixture");
}

template<typename Function>
void require_rejected(Function&& operation, const char* message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, message);
}

void test_loads_strict_little_endian_clip_and_preserves_pose_fields() {
    const std::filesystem::path path = fixture_path("offline-overlap-valid.bin");
    write_fixture(path);
    const interaction::offline_overlap::Clip clip =
        interaction::offline_overlap::Clip::load(path);
    require(clip.frames.size() == 80U, "wrong frame count");
    const interaction::Pose& pose = clip.frames[7];
    require(pose.positions[4].y == 713.0F, "position payload changed");
    require(pose.velocities[4].z == 2.0F, "velocity payload changed");
    require(pose.angular_velocities[4].x == 1.0F, "angular velocity payload changed");
    require(pose.rotations[4].w == 1.0F, "WXYZ rotation payload changed");
    require(pose.foot_contacts == std::array<uint8_t, 2>{1U, 0U},
            "foot contacts changed");
}

void test_rejects_trailing_malformed_nonfinite_and_nonunit_input() {
    FixtureOptions trailing;
    trailing.trailing_byte = true;
    FixtureOptions bad_frames;
    bad_frames.frame_count = 79U;
    FixtureOptions bad_bones;
    bad_bones.bone_count = 30U;
    FixtureOptions nonfinite;
    nonfinite.nonfinite_position = true;
    FixtureOptions nonunit;
    nonunit.nonunit_quaternion = true;
    const std::array<std::pair<const char*, FixtureOptions>, 5> cases = {{
        {"offline-overlap-trailing.bin", trailing},
        {"offline-overlap-frames.bin", bad_frames},
        {"offline-overlap-bones.bin", bad_bones},
        {"offline-overlap-nonfinite.bin", nonfinite},
        {"offline-overlap-nonunit.bin", nonunit},
    }};
    for (const auto& [name, options] : cases) {
        const std::filesystem::path path = fixture_path(name);
        write_fixture(path, options);
        require_rejected([&] {
            (void)interaction::offline_overlap::Clip::load(path);
        }, "malformed offline-overlap input was accepted");
    }
}

void test_rejects_wrong_skeleton_signature_and_order() {
    const std::filesystem::path signature_path = fixture_path("offline-overlap-signature.bin");
    FixtureOptions bad_signature;
    bad_signature.signature = std::string(64U, '0');
    write_fixture(signature_path, bad_signature);
    require_rejected([&] {
        (void)interaction::offline_overlap::Clip::load(signature_path);
    }, "wrong skeleton signature was accepted");

    const std::filesystem::path order_path = fixture_path("offline-overlap-order.bin");
    write_fixture(order_path);
    std::fstream stream(order_path, std::ios::binary | std::ios::in | std::ios::out);
    require(stream.is_open(), "could not corrupt skeleton order fixture");
    const std::streamoff first_name =
        static_cast<std::streamoff>(kMagic.size() + 6U * sizeof(uint32_t) + std::strlen(kSignature) + sizeof(uint32_t));
    stream.seekp(first_name);
    stream.put('X');
    stream.close();
    require_rejected([&] {
        (void)interaction::offline_overlap::Clip::load(order_path);
    }, "wrong skeleton order was accepted");
}

void test_player_advances_exactly_one_25hz_frame_and_can_loop_or_restart() {
    const std::filesystem::path path = fixture_path("offline-overlap-player.bin");
    write_fixture(path);
    const interaction::offline_overlap::Clip clip =
        interaction::offline_overlap::Clip::load(path);

    interaction::offline_overlap::Player once(clip, false);
    require(once.frame_index() == 0U, "player did not start at frame zero");
    for (size_t tick = 0U; tick < 79U; ++tick) once.advance_25hz();
    require(once.frame_index() == 79U, "player did not advance at 25 Hz");
    once.advance_25hz();
    require(once.finished(), "non-looping player did not finish");
    require(once.frame_index() == 79U, "finished player did not hold final frame");
    once.restart();
    require(!once.finished() && once.frame_index() == 0U,
            "restart did not restore the initial frame");

    interaction::offline_overlap::Player loop(clip, true);
    for (size_t tick = 0U; tick < 80U; ++tick) loop.advance_25hz();
    require(loop.frame_index() == 0U, "looping player did not wrap at frame 80");
}

void test_player_aligns_clip_root_to_live_pose_without_losing_travel() {
    const std::filesystem::path path = fixture_path("offline-overlap-aligned.bin");
    write_fixture(path);
    const interaction::offline_overlap::Clip clip =
        interaction::offline_overlap::Clip::load(path);
    interaction::offline_overlap::Player player(clip, false);
    interaction::Pose live{};
    live.positions[g1_skeleton::Simulation] = vec3(7.0F, 0.5F, -3.0F);
    live.rotations[g1_skeleton::Simulation] = quat(
        0.70710677F, 0.0F, 0.70710677F, 0.0F);

    player.restart_aligned_to(live);
    const interaction::Pose first = player.aligned_pose();
    require(
        length(first.positions[g1_skeleton::Simulation] -
               live.positions[g1_skeleton::Simulation]) < 1.0e-4F,
        "aligned player frame zero did not preserve the live root position");
    const quat aligned_rotation =
        first.rotations[g1_skeleton::Simulation];
    const quat live_rotation = live.rotations[g1_skeleton::Simulation];
    const float rotation_dot = std::abs(
        aligned_rotation.w * live_rotation.w +
        aligned_rotation.x * live_rotation.x +
        aligned_rotation.y * live_rotation.y +
        aligned_rotation.z * live_rotation.z);
    require(
        std::abs(rotation_dot - 1.0F) < 1.0e-4F,
        "aligned player frame zero did not preserve the live root rotation");

    const float source_travel = length(
        clip.frames[1].positions[g1_skeleton::Simulation] -
        clip.frames[0].positions[g1_skeleton::Simulation]);
    player.advance_25hz();
    const interaction::Pose second = player.aligned_pose();
    require(
        std::abs(length(
            second.positions[g1_skeleton::Simulation] -
            first.positions[g1_skeleton::Simulation]) - source_travel) < 1.0e-3F,
        "aligned player changed the clip's relative root travel");
}

void test_player_bridges_live_root_to_world_interaction_slot() {
    const std::filesystem::path path = fixture_path("offline-overlap-bridge.bin");
    write_fixture(path);
    const interaction::offline_overlap::Clip clip =
        interaction::offline_overlap::Clip::load(path);
    interaction::offline_overlap::Player player(clip, false);
    interaction::Pose live{};
    live.positions[g1_skeleton::Simulation] = vec3(2.0F, 0.25F, -1.0F);
    live.rotations[g1_skeleton::Simulation] = quat();
    const interaction::Transform destination{
        vec3(-4.0F, 0.25F, 8.0F),
        quat(0.70710677F, 0.0F, 0.70710677F, 0.0F)};

    player.restart_bridged_to(live, destination);
    const interaction::Pose first = player.aligned_pose();
    require(
        length(first.positions[g1_skeleton::Simulation] -
               live.positions[g1_skeleton::Simulation]) < 1.0e-4F,
        "bridged player changed the live entry root");
    for (size_t tick = 0U; tick < 79U; ++tick) player.advance_25hz();
    const interaction::Pose last = player.aligned_pose();
    require(
        length(last.positions[g1_skeleton::Simulation] -
               destination.position) < 1.0e-3F,
        "bridged player did not end at the world interaction slot");
    const quat end_rotation = last.rotations[g1_skeleton::Simulation];
    const float rotation_dot = std::abs(
        end_rotation.w * destination.rotation.w +
        end_rotation.x * destination.rotation.x +
        end_rotation.y * destination.rotation.y +
        end_rotation.z * destination.rotation.z);
    require(
        std::abs(rotation_dot - 1.0F) < 1.0e-4F,
        "bridged player did not end at the interaction-slot heading");
}

void test_player_maps_object_relative_clip_endpoint_through_world_frame() {
    const std::filesystem::path path = fixture_path("offline-overlap-object-frame.bin");
    write_fixture(path);
    const interaction::offline_overlap::Clip clip =
        interaction::offline_overlap::Clip::load(path);
    interaction::offline_overlap::Player player(clip, false);
    interaction::Pose live{};
    live.positions[g1_skeleton::Simulation] = vec3(1.0F, 0.0F, -2.0F);
    live.rotations[g1_skeleton::Simulation] = quat();
    const interaction::Transform object_yaw_world{
        vec3(0.25F, 0.5F, 3.0F),
        quat(0.9238795F, 0.0F, 0.3826834F, 0.0F)};
    const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const interaction::Transform expected = interaction::compose(
        object_yaw_world,
        interaction::Transform{
            clip.frames.back().positions[root],
            clip.frames.back().rotations[root]});

    player.restart_bridged_to_frame(live, object_yaw_world);
    for (size_t tick = 0U; tick < 79U; ++tick) player.advance_25hz();
    const interaction::Pose last = player.aligned_pose();
    require(
        length(last.positions[root] - expected.position) < 1.0e-3F,
        "object-relative endpoint was not mapped through the world frame");
    const quat actual_rotation = last.rotations[root];
    const float rotation_dot = std::abs(
        actual_rotation.w * expected.rotation.w +
        actual_rotation.x * expected.rotation.x +
        actual_rotation.y * expected.rotation.y +
        actual_rotation.z * expected.rotation.z);
    require(
        std::abs(rotation_dot - 1.0F) < 1.0e-4F,
        "object-relative endpoint heading was not mapped through the world frame");
}

void test_bridge_finishes_before_pickup_motion_begins() {
    const std::filesystem::path path = fixture_path("offline-overlap-settled-bridge.bin");
    write_fixture(path);
    const interaction::offline_overlap::Clip clip =
        interaction::offline_overlap::Clip::load(path);
    interaction::offline_overlap::Player player(clip, false);
    interaction::Pose live{};
    live.positions[g1_skeleton::Simulation] =
        clip.frames[0].positions[g1_skeleton::Simulation];
    live.rotations[g1_skeleton::Simulation] =
        clip.frames[0].rotations[g1_skeleton::Simulation];
    player.restart_bridged_to(
        live,
        interaction::Transform{vec3(-20.0F, 0.0F, 30.0F), quat()});
    for (size_t tick = 0U; tick < interaction::offline_overlap::kBridgeSettleFrame;
         ++tick) {
        player.advance_25hz();
    }
    const interaction::Pose settled = player.aligned_pose();
    player.advance_25hz();
    const interaction::Pose after = player.aligned_pose();
    const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const vec3 source_delta =
        clip.frames[interaction::offline_overlap::kBridgeSettleFrame + 1U]
                .positions[root] -
        clip.frames[interaction::offline_overlap::kBridgeSettleFrame]
                .positions[root];
    require(
        length((after.positions[root] - settled.positions[root]) - source_delta) <
            1.0e-3F,
        "bridge correction continued sliding during pickup motion");
}

void test_attached_object_follower_is_continuous_and_tracks_hand() {
    interaction::offline_overlap::AttachedObjectFollower follower;
    const interaction::Transform contact_hand{
        vec3(1.0F, 1.2F, -0.5F),
        quat(0.9238795F, 0.0F, 0.3826834F, 0.0F)};
    const interaction::Transform contact_object{
        vec3(1.1F, 1.0F, -0.45F),
        quat(0.9659258F, 0.0F, 0.2588190F, 0.0F)};
    follower.attach(contact_hand, contact_object);
    const interaction::Transform unchanged = follower.follow(contact_hand);
    require(
        length(unchanged.position - contact_object.position) < 1.0e-4F,
        "offline attachment moved the object at contact");
    const interaction::Transform moved_hand{
        vec3(2.0F, 1.5F, 0.25F),
        quat(0.70710677F, 0.0F, 0.70710677F, 0.0F)};
    const interaction::Transform expected = interaction::compose(
        moved_hand,
        interaction::compose(interaction::inverse(contact_hand), contact_object));
    const interaction::Transform followed = follower.follow(moved_hand);
    require(
        length(followed.position - expected.position) < 1.0e-4F,
        "offline attachment did not follow the hand");
    follower.reset();
    require(!follower.attached(), "offline attachment reset did not detach");
}

}  // namespace

int main() {
    test_loads_strict_little_endian_clip_and_preserves_pose_fields();
    test_rejects_trailing_malformed_nonfinite_and_nonunit_input();
    test_rejects_wrong_skeleton_signature_and_order();
    test_player_advances_exactly_one_25hz_frame_and_can_loop_or_restart();
    test_player_aligns_clip_root_to_live_pose_without_losing_travel();
    test_player_bridges_live_root_to_world_interaction_slot();
    test_player_maps_object_relative_clip_endpoint_through_world_frame();
    test_bridge_finishes_before_pickup_motion_begins();
    test_attached_object_follower_is_continuous_and_tracks_hand();
    return 0;
}
