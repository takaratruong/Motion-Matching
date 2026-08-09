#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "database.h"
#include "g1_skeleton.h"
#include "support_runtime.h"

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "support matching test failed: %s\n", message);
        std::exit(1);
    }
}

static uint32_t bits(float value)
{
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static size_t find_required(
    const std::string& source,
    const char* needle,
    const size_t start,
    const char* message)
{
    const size_t position = source.find(needle, start);
    check(position != std::string::npos, message);
    return position;
}

static size_t count_occurrences(
    const std::string& source, const char* needle)
{
    size_t count = 0;
    size_t position = 0;
    while ((position = source.find(needle, position)) != std::string::npos) {
        ++count;
        position += std::strlen(needle);
    }
    return count;
}

static void test_pure_matching_invariance()
{
    database db;
    db.bone_positions.resize(3, G1_BoneCount);
    db.features.resize(3, 31);
    db.features_offset.resize(31);
    db.features_scale.resize(31);
    db.range_starts.resize(1);
    db.range_stops.resize(1);
    db.terrain_features.resize(3, 4);
    db.features.zero();
    db.features_offset.zero();
    db.features_scale.set(1);
    db.features(0, 0) = 2;
    db.features(1, 0) = 0;
    db.features(2, 0) = 3;
    db.range_starts(0) = 0;
    db.range_stops(0) = 3;
    database_build_bounds(db);

    array1d<float> query(31);
    query.zero();
    int before = 0;
    float before_cost = FLT_MAX;
    database_search(before, before_cost, db, query, 0, 0, 1);

    array1d<vec3> pose(G1_BoneCount);
    array1d<vec3> retargeted(G1_BoneCount);
    pose.set(vec3());
    pose(G1_Hips) = vec3(0, 0.8f, 0);
    support_pose_apply(retargeted, pose, 0.36f);

    int after = 0;
    float after_cost = FLT_MAX;
    database_search(after, after_cost, db, query, 0, 0, 1);
    check(before == 1 && after == before, "selected frame invariant");
    check(bits(before_cost) == bits(after_cost),
          "selected cost bit invariant");
    check(retargeted(G1_Simulation).y == 0.36f &&
          bits(retargeted(G1_Hips).y) == bits(pose(G1_Hips).y),
          "only support transform changed");

    const vec3 character(1.0f, -0.0f, 2.0f);
    const vec3 simulation(4.0f, 19.0f, -3.0f);
    const vec3 velocity(0.4f, 11.0f, -0.2f);
    const vec3 adjusted = horizontal_adjust_character_position(
        character, simulation, 0.1f, 0.04f);
    const vec3 velocity_adjusted =
        horizontal_adjust_character_position_by_velocity(
            character, velocity, simulation, 0.5f, 0.1f, 0.04f);
    const vec3 clamped = horizontal_clamp_character_position(
        character, simulation, 0.15f);
    check(bits(adjusted.y) == bits(character.y) &&
          bits(velocity_adjusted.y) == bits(character.y) &&
          bits(clamped.y) == bits(character.y),
          "horizontal helpers preserve exact source Y bits");
    check(adjusted.y - character.y == 0.0f &&
          velocity_adjusted.y - character.y == 0.0f &&
          clamped.y - character.y == 0.0f,
          "horizontal helper Y displacements are zero");
}

static void test_controller_source_contract(const char* path)
{
    std::ifstream input(path);
    check(input.good(), "controller source opens");
    const std::string source(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());

    check(source.find("g1_runtime_step(") != std::string::npos,
          "controller delegates the ordinary matcher to the runtime kernel");
    check(source.find("const float dt = 1.0f / 60.0f;") !=
              std::string::npos &&
          source.find("SetTargetFPS(60);") != std::string::npos,
          "controller clock and viewer are fixed at 60 Hz");
    check(count_occurrences(
              source, "motion_manifest.output_fps") >= 2,
          "startup and UI feature builds consume the manifest rate");
    check(source.find("query_normalized_bits_hex") != std::string::npos,
          "ordinary runtime logs the shared normalized query snapshot");
    check(source.find("motion_manifest.flat_lmm_bundle") !=
              std::string::npos &&
          source.find("flat_scene_pack_build(") != std::string::npos &&
          source.find("database_load_matching_features_checked(") !=
              std::string::npos,
          "ordinary replay has an explicit fail-closed flat-bundle adapter");
    check(find_required(
              source, "motion_manifest_load_and_verify(", 0,
              "manifest loader exists") <
              find_required(
                  source, "g1_controller_state state;", 0,
                  "controller state allocation exists"),
          "manifest rate is rejected before controller/IK state allocation");
}

int main(int argc, char** argv)
{
    test_pure_matching_invariance();
    if (argc == 3 && std::strcmp(argv[1], "--controller") == 0) {
        test_controller_source_contract(argv[2]);
    } else {
        check(argc == 1, "usage: test_support_matching [--controller path]");
    }
    return 0;
}
