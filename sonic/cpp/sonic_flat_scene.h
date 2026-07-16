#pragma once

#include "scene_runtime.h"

#include <cstring>
#include <string>
#include <sys/stat.h>

static constexpr const char* SonicFlatSceneId = "sonic-flat-baseline";
static constexpr const char* SonicFlatSceneRegistrySchema =
    "mm-sonic-scene-registry/v1";
static constexpr const char* SonicFlatSceneRegistrySha256 =
    "85480e1e8a190d6b065749382172191850739c0aca8a8946182f97665d693d5b";

struct sonic_flat_scene_definition
{
    std::string scene_id;
    std::string kind;
    std::string coordinate_signature;
    bounds2 bounds;
    vec3 spawn_position;
    float spawn_yaw = 0.0f;
    float height = 0.0f;
    int walkability_class = 0;
    std::string registry_sha256;
    std::string registry_path;
};

static inline bool sonic_flat_scene_regular_file(
    const char* path, char* error, int capacity)
{
    const char* shown = path != nullptr ? path : "<null>";
    if (path == nullptr || path[0] == '\0') {
        return scene_error(error, capacity, "%s: invalid registry path", shown);
    }
    struct stat status = {};
    if (::lstat(path, &status) != 0) {
        return scene_error(
            error, capacity, "%s: cannot stat registry", shown);
    }
    if (S_ISLNK(status.st_mode)) {
        return scene_error(
            error, capacity, "%s: registry symlink is forbidden", shown);
    }
    if (!S_ISREG(status.st_mode)) {
        return scene_error(
            error, capacity, "%s: registry must be a regular file", shown);
    }
    return true;
}

static inline bool sonic_flat_scene_string_member(
    std::string& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    return scene_member_string(
        output, object, key, "flat scene registry", error, capacity);
}

static inline bool sonic_flat_scene_number_member(
    float& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr) {
        return scene_error(
            error,
            capacity,
            "flat scene registry is missing required key '%s'",
            key);
    }
    return scene_number_binary32(output, *value, key, error, capacity);
}

static inline bool sonic_flat_scene_definition_load(
    sonic_flat_scene_definition& output,
    const char* registry_path,
    char* error,
    int capacity)
{
    if (!sonic_flat_scene_regular_file(registry_path, error, capacity)) {
        return false;
    }

    std::string registry_sha256;
    json_value document;
    if (!sha256_file_hex(
            registry_sha256, registry_path, error, capacity)) {
        return false;
    }
    if (registry_sha256 != SonicFlatSceneRegistrySha256) {
        return scene_error(
            error,
            capacity,
            "%s: scene registry SHA-256 changed",
            registry_path);
    }
    if (!json_document_load(document, registry_path, error, capacity) ||
        !scene_exact_keys(
            document,
            {"schema", "scenes"},
            "flat scene registry",
            error,
            capacity)) {
        return false;
    }

    std::string schema;
    if (!sonic_flat_scene_string_member(
            schema, document, "schema", error, capacity) ||
        schema != SonicFlatSceneRegistrySchema) {
        return scene_error(
            error, capacity, "%s: scene registry schema changed", registry_path);
    }

    const json_value* scenes = json_member(document, "scenes");
    if (scenes == nullptr ||
        !scene_exact_keys(
            *scenes,
            {SonicFlatSceneId},
            "flat scene registry scenes",
            error,
            capacity)) {
        return false;
    }
    const json_value* entry = json_member(*scenes, SonicFlatSceneId);
    if (entry == nullptr ||
        !scene_exact_keys(
            *entry,
            {"kind",
             "coordinate_signature",
             "bounds_xz",
             "spawn_position_holden",
             "spawn_yaw_holden",
             "height_m",
             "walkability_class"},
            "flat scene registry entry",
            error,
            capacity)) {
        return false;
    }

    sonic_flat_scene_definition candidate;
    candidate.scene_id = SonicFlatSceneId;
    float bounds[4] = {};
    float spawn[3] = {};
    int classification = 0;
    const json_value* bounds_value = json_member(*entry, "bounds_xz");
    const json_value* spawn_value =
        json_member(*entry, "spawn_position_holden");
    const json_value* class_value = json_member(*entry, "walkability_class");
    if (!sonic_flat_scene_string_member(
            candidate.kind, *entry, "kind", error, capacity) ||
        !sonic_flat_scene_string_member(
            candidate.coordinate_signature,
            *entry,
            "coordinate_signature",
            error,
            capacity) ||
        bounds_value == nullptr ||
        !scene_binary32_array(
            bounds, 4, *bounds_value, "bounds_xz", error, capacity) ||
        spawn_value == nullptr ||
        !scene_binary32_array(
            spawn,
            3,
            *spawn_value,
            "spawn_position_holden",
            error,
            capacity) ||
        !sonic_flat_scene_number_member(
            candidate.spawn_yaw,
            *entry,
            "spawn_yaw_holden",
            error,
            capacity) ||
        !sonic_flat_scene_number_member(
            candidate.height, *entry, "height_m", error, capacity) ||
        class_value == nullptr ||
        !scene_number_int(
            classification,
            *class_value,
            "walkability_class",
            error,
            capacity)) {
        return false;
    }

    candidate.bounds = {bounds[0], bounds[1], bounds[2], bounds[3]};
    candidate.spawn_position = vec3(spawn[0], spawn[1], spawn[2]);
    candidate.walkability_class = classification;
    candidate.registry_sha256 = registry_sha256;
    candidate.registry_path = registry_path;
    if (candidate.kind != "analytic-flat" ||
        candidate.coordinate_signature != G1_RuntimeCoordinateSignature ||
        candidate.bounds.min_x != -10.0f ||
        candidate.bounds.min_z != -10.0f ||
        candidate.bounds.max_x != 10.0f ||
        candidate.bounds.max_z != 10.0f ||
        candidate.spawn_position.x != 0.0f ||
        candidate.spawn_position.y != 0.0f ||
        candidate.spawn_position.z != 0.0f ||
        candidate.spawn_yaw != 0.0f || candidate.height != 0.0f ||
        candidate.walkability_class != 1) {
        return scene_error(
            error,
            capacity,
            "%s: sonic-flat-baseline analytic values changed",
            registry_path);
    }
    output = candidate;
    return true;
}

static inline bool sonic_flat_scene_build(
    scene_pack& output,
    const sonic_flat_scene_definition& definition,
    const std::string& route_id,
    char* error,
    int capacity)
{
    if (definition.scene_id != SonicFlatSceneId || route_id.empty()) {
        return scene_error(
            error, capacity, "flat scene requires its exact ID and a reset route");
    }

    scene_pack candidate;
    candidate.metadata.id = definition.scene_id;
    candidate.metadata.label = "SONIC Flat Baseline";
    candidate.metadata.provenance_kind = definition.kind;
    candidate.metadata.coordinate_signature =
        definition.coordinate_signature;
    candidate.metadata.surface_signature = G1_RuntimeSurfaceSignature;
    candidate.metadata.playable_bounds = definition.bounds;
    candidate.metadata.lookahead_bounds = definition.bounds;
    candidate.metadata.spawn_position = definition.spawn_position;
    candidate.metadata.spawn_yaw = definition.spawn_yaw;
    candidate.metadata.heightfield_nx = 2;
    candidate.metadata.heightfield_nz = 2;
    candidate.metadata.walkability_nx = 2;
    candidate.metadata.walkability_nz = 2;
    candidate.metadata.heightfield_origin_x = definition.bounds.min_x;
    candidate.metadata.heightfield_origin_z = definition.bounds.min_z;
    candidate.metadata.heightfield_cell_size = 20.0f;
    candidate.metadata.heightfield_exterior_height = definition.height;
    candidate.metadata.heightfield.path = definition.registry_path;
    candidate.metadata.heightfield.schema = SonicFlatSceneRegistrySchema;
    candidate.metadata.heightfield.sha256 = definition.registry_sha256;
    candidate.metadata.heightfield.version = 1;
    candidate.metadata.mesh = candidate.metadata.heightfield;
    candidate.metadata.walkability = candidate.metadata.heightfield;
    candidate.metadata.mesh_bounds.minimum = {-10.0, 0.0, -10.0};
    candidate.metadata.mesh_bounds.maximum = {10.0, 0.0, 10.0};
    candidate.metadata.heightfield_bounds = candidate.metadata.mesh_bounds;
    scene_region certified;
    certified.id = "sonic-flat-baseline";
    certified.bounds = definition.bounds;
    candidate.metadata.certified_regions.push_back(certified);
    scene_route route;
    route.id = route_id;
    route.expected_outcome = "traverse";
    route.walkability_class = definition.walkability_class;
    route.landing_hold_seconds = 0.0f;
    route.waypoints_xz.push_back(std::make_pair(0.0f, 0.0f));
    route.waypoints_xz.push_back(std::make_pair(0.0f, 1.0f));
    candidate.metadata.routes.push_back(route);

    candidate.terrain.version = 2;
    candidate.terrain.nx = 2;
    candidate.terrain.nz = 2;
    candidate.terrain.origin_x = definition.bounds.min_x;
    candidate.terrain.origin_z = definition.bounds.min_z;
    candidate.terrain.cell_size = 20.0f;
    candidate.terrain.exterior_height = definition.height;
    candidate.terrain.heights.resize(4);
    candidate.terrain.heights.set(definition.height);
    candidate.walkability.nx = 2;
    candidate.walkability.nz = 2;
    candidate.walkability.cells.resize(4);
    candidate.walkability.cells.set(
        static_cast<uint8_t>(definition.walkability_class));
    candidate.scene_path = definition.registry_path;

    if (!terrain_heightfield_is_queryable(candidate.terrain) ||
        !walkability_grid_matches_heightfield(
            candidate.walkability, candidate.terrain)) {
        return scene_error(
            error, capacity, "constructed flat scene is not queryable");
    }
    scene_pack_swap(output, candidate);
    return true;
}

static inline bool sonic_flat_scene_load(
    scene_pack& output,
    sonic_flat_scene_definition* definition_output,
    const char* registry_path,
    const std::string& route_id,
    char* error,
    int capacity)
{
    sonic_flat_scene_definition definition;
    scene_pack candidate;
    if (!sonic_flat_scene_definition_load(
            definition, registry_path, error, capacity) ||
        !sonic_flat_scene_build(
            candidate, definition, route_id, error, capacity)) {
        return false;
    }
    scene_pack_swap(output, candidate);
    if (definition_output != nullptr) {
        *definition_output = definition;
    }
    return true;
}

static inline bool sonic_flat_scene_query(
    float& height,
    int& walkability_class,
    const scene_pack& scene,
    float x,
    float z)
{
    if (scene.metadata.id != SonicFlatSceneId ||
        !scene_inside(scene.metadata.playable_bounds, x, z) ||
        !terrain_heightfield_is_queryable(scene.terrain) ||
        !walkability_grid_matches_heightfield(
            scene.walkability, scene.terrain)) {
        return false;
    }
    const float candidate_height = heightfield_sample_v2(scene.terrain, x, z);
    const int candidate_class =
        walkability_class_at(scene.walkability, scene.terrain, x, z);
    if (!terrain_float_is_finite(candidate_height) || candidate_class != 1) {
        return false;
    }
    height = candidate_height;
    walkability_class = candidate_class;
    return true;
}
