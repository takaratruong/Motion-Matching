#include "sonic/cpp/sonic_flat_scene.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

#include <unistd.h>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "sonic flat scene test failed: %s\n", message);
        std::exit(1);
    }
}

static void test_exact_registry_and_analytic_pack()
{
    char error[1024] = {};
    scene_pack scene;
    sonic_flat_scene_definition definition;
    check(
        sonic_flat_scene_load(
            scene,
            &definition,
            "sonic/configs/scene_registry.json",
            "flat-12s",
            error,
            static_cast<int>(sizeof(error))),
        error);

    check(definition.scene_id == "sonic-flat-baseline", "scene ID is exact");
    check(definition.kind == "analytic-flat", "kind is exact");
    check(
        definition.coordinate_signature ==
            "holden-y-up-right-handed-forward-plus-z",
        "coordinate signature is exact");
    check(
        definition.registry_sha256 ==
            "85480e1e8a190d6b065749382172191850739c0aca8a8946182f97665d693d5b",
        "registry bytes are exact");
    check(
        definition.bounds.min_x == -10.0f &&
            definition.bounds.min_z == -10.0f &&
            definition.bounds.max_x == 10.0f &&
            definition.bounds.max_z == 10.0f,
        "flat bounds are exact");
    check(
        definition.spawn_position.x == 0.0f &&
            definition.spawn_position.y == 0.0f &&
            definition.spawn_position.z == 0.0f &&
            definition.spawn_yaw == 0.0f,
        "flat spawn is exact");
    check(
        definition.height == 0.0f && definition.walkability_class == 1,
        "flat height and certified class are exact");

    check(scene.metadata.id == definition.scene_id, "pack uses registry ID");
    check(scene.metadata.routes.size() == 1u, "one reset route is staged");
    check(
        scene.metadata.routes[0].id == "flat-12s",
        "the request route identity is retained");
    check(
        scene.terrain.version == 2 && scene.terrain.nx == 2 &&
            scene.terrain.nz == 2 && scene.terrain.origin_x == -10.0f &&
            scene.terrain.origin_z == -10.0f &&
            scene.terrain.cell_size == 20.0f &&
            scene.terrain.exterior_height == 0.0f,
        "analytic heightfield has the exact inclusive domain");
    check(
        scene.walkability.nx == 2 && scene.walkability.nz == 2 &&
            scene.walkability.cells.size == 4,
        "analytic walkability has matching shape");
    for (int index = 0; index < 4; ++index) {
        check(scene.terrain.heights(index) == 0.0f, "every node is zero");
        check(scene.walkability.cells(index) == 1, "every node is certified");
    }

    const float probes[][2] = {
        {-10.0f, -10.0f}, {-10.0f, 10.0f}, {10.0f, -10.0f},
        {10.0f, 10.0f}, {0.0f, 0.0f}, {-3.25f, 7.75f},
    };
    for (const auto& probe : probes) {
        float height = std::numeric_limits<float>::quiet_NaN();
        int classification = -1;
        check(
            sonic_flat_scene_query(
                height, classification, scene, probe[0], probe[1]),
            "in-bounds query succeeds");
        check(
            std::isfinite(height) && height == 0.0f && classification == 1,
            "in-bounds query is finite, zero, and certified");
    }

    const float outside[][2] = {
        {-10.0001f, 0.0f}, {10.0001f, 0.0f},
        {0.0f, -10.0001f}, {0.0f, 10.0001f},
        {std::numeric_limits<float>::infinity(), 0.0f},
        {std::numeric_limits<float>::quiet_NaN(), 0.0f},
    };
    for (const auto& probe : outside) {
        float height = 17.0f;
        int classification = 17;
        check(
            !sonic_flat_scene_query(
                height, classification, scene, probe[0], probe[1]),
            "out-of-bounds or non-finite query is rejected");
    }
}

static void test_full_twelve_second_centerline_remains_queryable()
{
    char error[1024] = {};
    scene_pack scene;
    check(
        sonic_flat_scene_load(
            scene,
            nullptr,
            "sonic/configs/scene_registry.json",
            "flat-12s",
            error,
            static_cast<int>(sizeof(error))),
        error);

    float x = 0.0f;
    float z = 0.0f;
    for (int frame = 0; frame <= 300; ++frame) {
        float height = 1.0f;
        int classification = 0;
        check(
            sonic_flat_scene_query(height, classification, scene, x, z),
            "every 25 Hz centerline boundary stays in bounds");
        check(
            std::isfinite(height) && height == 0.0f && classification == 1,
            "every centerline boundary is finite and certified");
        if (frame >= 50 && frame < 150) {
            z += 0.5f * 0.04f;
        } else if (frame >= 150 && frame < 250) {
            const float alpha = static_cast<float>(frame - 150) / 100.0f;
            const float heading = alpha * 0.7853981633974483f;
            x += std::sin(heading) * 0.5f * 0.04f;
            z += std::cos(heading) * 0.5f * 0.04f;
        }
    }
}

static void test_registry_mutation_and_symlink_are_rejected()
{
    const std::string mutated = "/tmp/test_sonic_flat_scene_mutated.json";
    {
        std::ofstream output(mutated.c_str(), std::ios::binary | std::ios::trunc);
        output << "{\"schema\":\"mm-sonic-scene-registry/v1\","
                  "\"scenes\":{\"sonic-flat-baseline\":{"
                  "\"kind\":\"analytic-flat\","
                  "\"coordinate_signature\":"
                  "\"holden-y-up-right-handed-forward-plus-z\","
                  "\"bounds_xz\":[-10,-10,10,10],"
                  "\"spawn_position_holden\":[0,0,0],"
                  "\"spawn_yaw_holden\":0,\"height_m\":0,"
                  "\"walkability_class\":1}}}";
    }
    char error[1024] = {};
    scene_pack scene;
    check(
        !sonic_flat_scene_load(
            scene,
            nullptr,
            mutated.c_str(),
            "flat-12s",
            error,
            static_cast<int>(sizeof(error))),
        "semantically equal mutated registry bytes are rejected");

    const std::string link = "/tmp/test_sonic_flat_scene_registry_link.json";
    ::unlink(link.c_str());
    check(
        ::symlink(
            "sonic/configs/scene_registry.json",
            link.c_str()) == 0,
        "registry symlink fixture is created");
    error[0] = '\0';
    check(
        !sonic_flat_scene_load(
            scene,
            nullptr,
            link.c_str(),
            "flat-12s",
            error,
            static_cast<int>(sizeof(error))),
        "registry symlink is rejected");
    ::unlink(link.c_str());
    ::unlink(mutated.c_str());
}

int main()
{
    test_exact_registry_and_analytic_pack();
    test_full_twelve_second_centerline_remains_queryable();
    test_registry_mutation_and_symlink_are_rejected();
    std::puts("SONIC flat scene tests passed");
    return 0;
}
