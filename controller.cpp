#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wenum-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "raylib.h"
#include "raymath.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "common.h"
#include "vec.h"
#include "quat.h"
#include "spring.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "array.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "character.h"
#include "scene_runtime.h"
#include "motion_bank_runtime.h"
#include "route_runtime.h"
#include "support_runtime.h"
#include "g1_controller_state.h"
#include "g1_ik.h"
#include "g1_ik_runtime.h"
#include "g1_runtime_diagnostics.h"
#include "scene_switch.h"
#include "motion_match_log.h"
#include "cleanup_runtime.h"
#include "g1_mesh_renderer.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "nnet.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "lmm.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <cstdlib>
#include <initializer_list>
#include <functional>

//--------------------------------------

static inline Vector3 to_Vector3(vec3 v)
{
    return Vector3{ v.x, v.y, v.z };
}

static bool g1_parse_terrain_weight(
    float& weight, char* error, const int error_capacity)
{
    const char* text = getenv("MM_TERRAIN_WEIGHT");
    if (text == NULL) return true;

    errno = 0;
    char* end = NULL;
    const float parsed = strtof(text, &end);
    if (text[0] == '\0' || end == text || *end != '\0' || errno == ERANGE ||
        !terrain_float_is_finite(parsed) || parsed < 0.0f || parsed > 10.0f)
    {
        return g1_error(
            error,
            error_capacity,
            "MM_TERRAIN_WEIGHT must be a finite number in [0,10], got '%s'",
            text);
    }
    weight = parsed;
    return true;
}

static bool g1_parse_test_frames(
    int& frame_limit, char* error, const int error_capacity)
{
    const char* text = getenv("MM_TEST_FRAMES");
    if (text == NULL) return true;

    errno = 0;
    char* end = NULL;
    const long parsed = strtol(text, &end, 10);
    if (text[0] == '\0' || end == text || *end != '\0' || errno == ERANGE ||
        parsed <= 0 || parsed > INT_MAX)
    {
        return g1_error(
            error,
            error_capacity,
            "MM_TEST_FRAMES must be a positive integer, got '%s'",
            text);
    }
    frame_limit = static_cast<int>(parsed);
    return true;
}

enum g1_test_mode
{
    G1_TestLive,
    G1_TestSequential,
    G1_TestFlat,
    G1_TestTerrain,
    G1_TestRoute,
    G1_TestSceneCycle,
};

struct g1_test_config
{
    g1_test_mode mode = G1_TestLive;
    const char* name = "live";
    const char* route = "manual";
    int frame_limit = 0;
    int scene_dwell_frames = 25;
};

static bool g1_parse_scene_dwell_frames(
    int& dwell_frames, char* error, const int error_capacity)
{
    const char* text = getenv("MM_SCENE_DWELL_FRAMES");
    if (text == NULL) return true;

    errno = 0;
    char* end = NULL;
    const long parsed = strtol(text, &end, 10);
    if (text[0] == '\0' || end == text || *end != '\0' || errno == ERANGE ||
        parsed < 1 || parsed > 10000)
    {
        return g1_error(
            error,
            error_capacity,
            "MM_SCENE_DWELL_FRAMES must be an integer in [1,10000], got '%s'",
            text);
    }
    dwell_frames = static_cast<int>(parsed);
    return true;
}

static bool g1_parse_test_config(
    g1_test_config& out, char* error, const int error_capacity)
{
    const char* mode = getenv("MM_TEST_MODE");
    const char* route = getenv("MM_TEST_ROUTE");
    const char* dwell = getenv("MM_SCENE_DWELL_FRAMES");
    if (mode != NULL) {
        if (strcmp(mode, "sequential") == 0) {
            out.mode = G1_TestSequential; out.name = mode;
            out.route = "curb-forward";
        } else if (strcmp(mode, "flat") == 0) {
            out.mode = G1_TestFlat; out.name = mode;
            out.route = "curb-forward";
        } else if (strcmp(mode, "terrain") == 0) {
            out.mode = G1_TestTerrain; out.name = mode;
            out.route = "curb-forward";
        } else if (strcmp(mode, "route") == 0) {
            out.mode = G1_TestRoute; out.name = mode;
            if (route == NULL || route[0] == '\0') {
                return g1_error(
                    error, error_capacity,
                    "MM_TEST_ROUTE is required for route mode");
            }
            if (dwell != NULL) {
                return g1_error(
                    error, error_capacity,
                    "MM_SCENE_DWELL_FRAMES is invalid for route mode");
            }
            out.route = route;
        } else if (strcmp(mode, "scene-cycle") == 0) {
            out.mode = G1_TestSceneCycle; out.name = mode;
            out.route = "";
            if (route != NULL) {
                return g1_error(
                    error, error_capacity,
                    "MM_TEST_ROUTE is invalid for scene-cycle mode");
            }
            if (!g1_parse_scene_dwell_frames(
                    out.scene_dwell_frames, error, error_capacity))
            {
                return false;
            }
        } else {
            return g1_error(error, error_capacity,
                "MM_TEST_MODE must be sequential, flat, terrain, route, or "
                "scene-cycle, got '%s'",
                mode);
        }
        if (out.mode != G1_TestRoute &&
            out.mode != G1_TestSceneCycle && route != NULL)
        {
            return g1_error(
                error, error_capacity,
                "MM_TEST_ROUTE is valid only for route mode");
        }
        if (out.mode != G1_TestSceneCycle &&
            out.mode != G1_TestRoute && dwell != NULL)
        {
            return g1_error(
                error, error_capacity,
                "MM_SCENE_DWELL_FRAMES is valid only for scene-cycle mode");
        }
    } else {
        if (route != NULL) {
            return g1_error(
                error, error_capacity,
                "MM_TEST_ROUTE requires MM_TEST_MODE=route");
        }
        if (dwell != NULL) {
            return g1_error(
                error, error_capacity,
                "MM_SCENE_DWELL_FRAMES requires MM_TEST_MODE=scene-cycle");
        }
    }
    if (!g1_parse_test_frames(out.frame_limit, error, error_capacity)) {
        return false;
    }
    if (out.mode != G1_TestLive && out.frame_limit <= 0) {
        return g1_error(
            error, error_capacity,
            "MM_TEST_FRAMES must be set to a positive integer for %s mode",
            out.name);
    }
    if (out.mode == G1_TestSceneCycle &&
        out.frame_limit < 14 * out.scene_dwell_frames)
    {
        return g1_error(
            error, error_capacity,
            "MM_TEST_FRAMES must be at least 14*MM_SCENE_DWELL_FRAMES "
            "(%d) for scene-cycle mode",
            14 * out.scene_dwell_frames);
    }
    return true;
}

static motion_match_pose_diagnostic g1_pose_diagnostic(
    const slice1d<vec3> local_positions,
    const slice1d<quat> local_rotations,
    const slice1d<int> parents,
    const heightfield& terrain)
{
    array1d<vec3> positions(local_positions.size);
    array1d<quat> rotations(local_rotations.size);
    forward_kinematics_full(
        positions, rotations, local_positions, local_rotations, parents);
    motion_match_pose_diagnostic out;
    out.hips_y = positions(G1_Hips).y;
    out.hips_clearance = positions(G1_Hips).y - heightfield_sample_v2(
        terrain, positions(G1_Hips).x, positions(G1_Hips).z);
    out.left_toe_clearance = positions(G1_LeftToe).y - heightfield_sample_v2(
        terrain, positions(G1_LeftToe).x, positions(G1_LeftToe).z);
    out.right_toe_clearance = positions(G1_RightToe).y - heightfield_sample_v2(
        terrain, positions(G1_RightToe).x, positions(G1_RightToe).z);
    out.minimum_clearance = out.hips_clearance;
    const int probes[] = {
        G1_LeftKnee, G1_RightKnee, G1_LeftAnkle, G1_RightAnkle,
        G1_LeftToe, G1_RightToe
    };
    for (int i = 0; i < 6; ++i) {
        const vec3 p = positions(probes[i]);
        out.minimum_clearance = minf(
            out.minimum_clearance,
            p.y - heightfield_sample_v2(terrain, p.x, p.z));
    }
    return out;
}

static bool g1_pose_diagnostic_is_finite(
    const motion_match_pose_diagnostic& diagnostic)
{
    return terrain_float_is_finite(diagnostic.hips_y) &&
           terrain_float_is_finite(diagnostic.hips_clearance) &&
           terrain_float_is_finite(diagnostic.left_toe_clearance) &&
           terrain_float_is_finite(diagnostic.right_toe_clearance) &&
           terrain_float_is_finite(diagnostic.minimum_clearance);
}

template<typename T>
static bool g1_validate_animation_shape(
    const array2d<T>& values,
    const int frames,
    const int bones,
    const char* name,
    char* error,
    const int error_capacity)
{
    if (values.rows != frames || values.cols != bones || values.data == NULL)
    {
        return g1_error(
            error,
            error_capacity,
            "G1 database %s shape mismatch: expected %dx%d, got %dx%d",
            name,
            frames,
            bones,
            values.rows,
            values.cols);
    }
    return true;
}

static bool g1_database_validate(
    const database& db, char* error, const int error_capacity)
{
    const int frames = db.nframes();
    const int bones = db.nbones();
    if (frames <= 0)
    {
        return g1_error(error, error_capacity, "G1 database has no frames");
    }
    if (!g1_validate_animation_shape(
            db.bone_positions,
            frames,
            bones,
            "bone_positions",
            error,
            error_capacity) ||
        !g1_validate_animation_shape(
            db.bone_velocities,
            frames,
            bones,
            "bone_velocities",
            error,
            error_capacity) ||
        !g1_validate_animation_shape(
            db.bone_rotations,
            frames,
            bones,
            "bone_rotations",
            error,
            error_capacity) ||
        !g1_validate_animation_shape(
            db.bone_angular_velocities,
            frames,
            bones,
            "bone_angular_velocities",
            error,
            error_capacity))
    {
        return false;
    }
    if (db.contact_states.rows != frames || db.contact_states.cols != 2 ||
        db.contact_states.data == NULL)
    {
        return g1_error(
            error,
            error_capacity,
            "G1 database contact shape mismatch: expected %dx2, got %dx%d",
            frames,
            db.contact_states.rows,
            db.contact_states.cols);
    }
    if (db.range_starts.size <= 0 ||
        db.range_stops.size != db.range_starts.size ||
        db.range_starts.data == NULL || db.range_stops.data == NULL)
    {
        return g1_error(
            error,
            error_capacity,
            "G1 database range arrays must be nonempty and equal-sized");
    }

    int expected_start = 0;
    for (int range = 0; range < db.nranges(); ++range)
    {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        if (start != expected_start || stop <= start || stop > frames)
        {
            return g1_error(
                error,
                error_capacity,
                "G1 database range %d is not contiguous/in-bounds: "
                "expected start %d, got [%d,%d) for %d frames",
                range,
                expected_start,
                start,
                stop,
                frames);
        }
        expected_start = stop;
    }
    if (expected_start != frames)
    {
        return g1_error(
            error,
            error_capacity,
            "G1 database ranges stop at %d instead of covering %d frames",
            expected_start,
            frames);
    }
    return true;
}

static bool g1_feature_value_is_safe(const float value)
{
    const uint32_t magnitude = feature_float_bits(value) & UINT32_C(0x7fffffff);
    return feature_float_is_finite(value) && magnitude != UINT32_C(0x7f7fffff);
}

static bool g1_matching_features_validate(
    const database& db, char* error, const int error_capacity)
{
    const int expected_features = DATABASE_G1_MATCHING_FEATURES;
    if (db.features.rows != db.nframes() ||
        db.features.cols != expected_features || db.features.data == NULL ||
        db.features_offset.size != expected_features ||
        db.features_scale.size != expected_features ||
        db.features_offset.data == NULL || db.features_scale.data == NULL)
    {
        return g1_error(
            error,
            error_capacity,
            "G1 matching feature build failed: expected %dx%d features, got %dx%d",
            db.nframes(),
            expected_features,
            db.features.rows,
            db.features.cols);
    }

    for (int feature = 0; feature < expected_features; ++feature)
    {
        if (!g1_feature_value_is_safe(db.features_offset(feature)) ||
            !feature_float_is_positive_finite(db.features_scale(feature)))
        {
            return g1_error(
                error,
                error_capacity,
                "G1 matching feature %d has invalid offset/scale",
                feature);
        }
    }
    for (int value = 0; value < db.features.rows * db.features.cols; ++value)
    {
        if (!g1_feature_value_is_safe(db.features.data[value]))
        {
            return g1_error(
                error,
                error_capacity,
                "G1 matching feature row payload is invalid at value %d",
                value);
        }
    }

    const int small_rows =
        (db.nframes() + BOUND_SM_SIZE - 1) / BOUND_SM_SIZE;
    const int large_rows =
        (db.nframes() + BOUND_LR_SIZE - 1) / BOUND_LR_SIZE;
    const array2d<float>* bounds[4] = {
        &db.bound_sm_min, &db.bound_sm_max, &db.bound_lr_min, &db.bound_lr_max
    };
    const int expected_rows[4] = {
        small_rows, small_rows, large_rows, large_rows
    };
    for (int bound = 0; bound < 4; ++bound)
    {
        if (bounds[bound]->rows != expected_rows[bound] ||
            bounds[bound]->cols != expected_features ||
            bounds[bound]->data == NULL)
        {
            return g1_error(
                error,
                error_capacity,
                "G1 matching bound %d shape mismatch: expected %dx%d, got %dx%d",
                bound,
                expected_rows[bound],
                expected_features,
                bounds[bound]->rows,
                bounds[bound]->cols);
        }
        for (int value = 0; value < bounds[bound]->rows * bounds[bound]->cols;
             ++value)
        {
            if (!g1_feature_value_is_safe(bounds[bound]->data[value]))
            {
                return g1_error(
                    error,
                    error_capacity,
                    "G1 matching bound %d has invalid value at %d",
                    bound,
                    value);
            }
        }
    }
    for (int value = 0; value < small_rows * expected_features; ++value)
    {
        if (db.bound_sm_min.data[value] > db.bound_sm_max.data[value])
        {
            return g1_error(
                error, error_capacity, "G1 small matching bounds are inverted");
        }
    }
    for (int value = 0; value < large_rows * expected_features; ++value)
    {
        if (db.bound_lr_min.data[value] > db.bound_lr_max.data[value])
        {
            return g1_error(
                error, error_capacity, "G1 large matching bounds are inverted");
        }
    }
    return true;
}

static bool g1_predicted_motion_request_masks(
    uint16_t& direction_mask,
    uint8_t& speed_mask,
    const slice1d<vec3> predicted_positions,
    const slice1d<quat> predicted_headings,
    char* error,
    const int error_capacity)
{
    if (predicted_positions.size < 2 ||
        predicted_positions.size != predicted_headings.size ||
        predicted_positions.data == NULL || predicted_headings.data == NULL)
    {
        return g1_error(
            error, error_capacity,
            "predicted motion request requires aligned trajectory samples");
    }

    const int horizon = predicted_positions.size - 1;
    const vec3 start = predicted_positions(0);
    const vec3 stop = predicted_positions(horizon);
    if (!terrain_float_is_finite(start.x) ||
        !terrain_float_is_finite(start.z) ||
        !terrain_float_is_finite(stop.x) ||
        !terrain_float_is_finite(stop.z))
    {
        return g1_error(
            error, error_capacity,
            "predicted motion request contains a non-finite position");
    }

    const double dx = static_cast<double>(stop.x) -
        static_cast<double>(start.x);
    const double dz = static_cast<double>(stop.z) -
        static_cast<double>(start.z);
    const double distance = hypot(dx, dz);
    if (!terrain_double_is_finite(distance))
    {
        return g1_error(
            error, error_capacity,
            "predicted motion request displacement is non-finite");
    }

    const vec3 start_heading = terrain_centerline_flattened_heading(
        predicted_headings(0), vec3(0.0f, 0.0f, 1.0f));
    const vec3 horizon_heading = terrain_centerline_flattened_heading(
        predicted_headings(horizon), start_heading);
    const double heading = atan2(
        static_cast<double>(start_heading.x),
        static_cast<double>(start_heading.z));
    const double future_heading = atan2(
        static_cast<double>(horizon_heading.x),
        static_cast<double>(horizon_heading.z));
    if (!terrain_double_is_finite(heading) ||
        !terrain_double_is_finite(future_heading))
    {
        return g1_error(
            error, error_capacity,
            "predicted motion request heading is non-finite");
    }

    static const double pi = 3.14159265358979323846264338327950288;
    static const double moving_minimum_m = 0.08;
    static const double low_speed_maximum_m = 0.12;
    static const double sector_half_width = 27.5 * pi / 180.0;
    static const double sector_tolerance = 8.0e-15;
    static const double lateral_heading_limit = 15.0 * pi / 180.0;
    static const uint16_t moving_bits[8] = {
        MOTION_DIRECTION_FORWARD,
        MOTION_DIRECTION_FORWARD_RIGHT,
        MOTION_DIRECTION_RIGHT,
        MOTION_DIRECTION_BACK_RIGHT,
        MOTION_DIRECTION_BACKWARD,
        MOTION_DIRECTION_BACK_LEFT,
        MOTION_DIRECTION_LEFT,
        MOTION_DIRECTION_FORWARD_LEFT,
    };

    direction_mask = MOTION_DIRECTION_IDLE;
    if (distance >= moving_minimum_m)
    {
        const double sine = sin(heading);
        const double cosine = cos(heading);
        const double local_right = dx * cosine - dz * sine;
        const double local_forward = dx * sine + dz * cosine;
        const double travel_angle = atan2(local_right, local_forward);
        direction_mask = 0;
        for (int sector = 0; sector < 8; ++sector)
        {
            const double center = static_cast<double>(sector) * pi / 4.0;
            const double angular_distance = fabs(
                remainder(travel_angle - center, 2.0 * pi));
            if (angular_distance <= sector_half_width + sector_tolerance)
                direction_mask |= moving_bits[sector];
        }

        const double heading_change = fabs(
            remainder(future_heading - heading, 2.0 * pi));
        if (heading_change > lateral_heading_limit)
        {
            direction_mask &= static_cast<uint16_t>(
                MOTION_DIRECTION_FORWARD | MOTION_DIRECTION_BACKWARD);
        }
        if (direction_mask == 0) direction_mask = MOTION_DIRECTION_IDLE;
    }

    speed_mask = 0;
    if (distance <= low_speed_maximum_m) speed_mask |= MOTION_SPEED_LOW;
    if (distance >= moving_minimum_m) speed_mask |= MOTION_SPEED_MOVING;
    if (!motion_index_direction_is_valid(direction_mask) ||
        !motion_index_speed_is_valid(speed_mask))
    {
        return g1_error(
            error, error_capacity,
            "predicted motion request produced invalid masks");
    }
    return true;
}

static int g1_active_range(const database& db, const int frame)
{
    for (int range = 0; range < db.nranges(); ++range)
    {
        if (frame >= db.range_starts(range) && frame < db.range_stops(range))
        {
            return range;
        }
    }
    return -1;
}

//-------------------------------------- MM_DISCRETE instrumentation
#ifdef MM_DISCRETE
#include <cstdio>
#include <cstdlib>
int          g_frame = 0;
static bool  g_force_strafe = false;
static FILE* g_log = nullptr;
// yaw (deg) of the facing direction produced by a quat, matches how
// desired_rotation is derived (atan2 of a rotated forward vector)
static inline float dbg_yaw_deg(quat q)
{
    vec3 f = quat_mul_vec3(q, vec3(0.0f, 0.0f, 1.0f));
    return atan2f(f.x, f.z) * 180.0f / PIf;
}
static inline float dbg_quat_angle_deg(quat q)
{
    q = quat_abs(q);
    return 2.0f * acosf(clampf(q.w, -1.0f, 1.0f)) * 180.0f / PIf;
}
static inline float dbg_angle_between_deg(quat a, quat b)
{
    return quat_angle_between(a, b) * 180.0f / PIf;
}
#endif

//--------------------------------------

// Perform linear blend skinning and copy 
// result into mesh data. Update and upload 
// deformed vertex positions and normals to GPU
void deform_character_mesh(
  Mesh& mesh, 
  const character& c,
  const slice1d<vec3> bone_anim_positions,
  const slice1d<quat> bone_anim_rotations,
  const slice1d<int> bone_parents)
{
    (void)bone_parents;

    linear_blend_skinning_positions(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.vertices),
        c.positions,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_positions,
        c.bone_rest_rotations,
        bone_anim_positions,
        bone_anim_rotations);
    
    linear_blend_skinning_normals(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.normals),
        c.normals,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_rotations,
        bone_anim_rotations);
    
    UpdateMeshBuffer(mesh, 0, mesh.vertices, mesh.vertexCount * 3 * sizeof(float), 0);
    UpdateMeshBuffer(mesh, 2, mesh.normals, mesh.vertexCount * 3 * sizeof(float), 0);
}

Mesh make_character_mesh(character& c)
{
    Mesh mesh{};
    
    mesh.vertexCount = c.positions.size;
    mesh.triangleCount = c.triangles.size / 3;
    mesh.vertices = (float*)MemAlloc(c.positions.size * 3 * sizeof(float));
    mesh.texcoords = (float*)MemAlloc(c.texcoords.size * 2 * sizeof(float));
    mesh.normals = (float*)MemAlloc(c.normals.size * 3 * sizeof(float));
    mesh.indices = (unsigned short*)MemAlloc(c.triangles.size * sizeof(unsigned short));
    
    memcpy(mesh.vertices, c.positions.data, c.positions.size * 3 * sizeof(float));
    memcpy(mesh.texcoords, c.texcoords.data, c.texcoords.size * 2 * sizeof(float));
    memcpy(mesh.normals, c.normals.data, c.normals.size * 3 * sizeof(float));
    memcpy(mesh.indices, c.triangles.data, c.triangles.size * sizeof(unsigned short));
    
    UploadMesh(&mesh, true);
    
    return mesh;
}

//--------------------------------------

// Basic functionality to get gamepad input including deadzone and 
// squaring of the stick location to increase sensitivity. To make 
// all the other code that uses this easier, we assume stick is 
// oriented on floor (i.e. y-axis is zero)

enum
{
    GAMEPAD_PLAYER = 0,
};

enum
{
    GAMEPAD_STICK_LEFT,
    GAMEPAD_STICK_RIGHT,
};

vec3 gamepad_get_stick(int stick, const float deadzone = 0.2f)
{
    float gamepadx = GetGamepadAxisMovement(GAMEPAD_PLAYER, stick == GAMEPAD_STICK_LEFT ? GAMEPAD_AXIS_LEFT_X : GAMEPAD_AXIS_RIGHT_X);
    float gamepady = GetGamepadAxisMovement(GAMEPAD_PLAYER, stick == GAMEPAD_STICK_LEFT ? GAMEPAD_AXIS_LEFT_Y : GAMEPAD_AXIS_RIGHT_Y);

#ifdef MM_AUTODRIVE
    // self-driving test mode: walk in a slowly-turning circle
    if (stick == GAMEPAD_STICK_LEFT)
    {
        float t = (float)GetTime();
        return vec3(0.7f*sinf(0.4f*t), 0.0f, -0.7f*cosf(0.4f*t));
    }
#endif
#ifdef MM_DISCRETE
    // discrete-keyboard test mode. Scripted phases mimic a user mashing keys:
    //   frames   0-119 : hold W (forward, -Z)
    //   frames 120-179 : hold S (backward, +Z)   <- 180 deg move reversal
    //   frames 180-239 : hold A (strafe-left, -X)
    //   frames 240-299 : hold D (strafe-right, +X)
    //   frames 300-399 : hold W again (forward)
    // Heading is ALSO snapped via camera_azimuth in the main loop.
    if (stick == GAMEPAD_STICK_LEFT)
    {
        int f = g_frame;
        if      (f < 120) return vec3( 0.0f, 0.0f, -0.9f); // W forward
        else if (f < 180) return vec3( 0.0f, 0.0f, +0.9f); // S backward (180 flip)
        else if (f < 240) return vec3(-0.9f, 0.0f,  0.0f); // A left
        else if (f < 300) return vec3(+0.9f, 0.0f,  0.0f); // D right
        else              return vec3( 0.0f, 0.0f, -0.9f); // W forward
    }
    else
    {
        return vec3(); // right stick unused; azimuth is snapped directly
    }
#endif
    // keyboard fallback: WASD -> left stick (move), arrows -> right stick (camera)
    if (stick == GAMEPAD_STICK_LEFT)
    {
        if (IsKeyDown(KEY_A)) gamepadx -= 1.0f;
        if (IsKeyDown(KEY_D)) gamepadx += 1.0f;
        if (IsKeyDown(KEY_W)) gamepady -= 1.0f;
        if (IsKeyDown(KEY_S)) gamepady += 1.0f;
    }
    else
    {
        if (IsKeyDown(KEY_LEFT))  gamepadx -= 1.0f;
        if (IsKeyDown(KEY_RIGHT)) gamepadx += 1.0f;
        if (IsKeyDown(KEY_UP))    gamepady -= 1.0f;
        if (IsKeyDown(KEY_DOWN))  gamepady += 1.0f;
    }
    float gamepadmag = sqrtf(gamepadx*gamepadx + gamepady*gamepady);
    
    if (gamepadmag > deadzone)
    {
        float gamepaddirx = gamepadx / gamepadmag;
        float gamepaddiry = gamepady / gamepadmag;
        float gamepadclippedmag = gamepadmag > 1.0f ? 1.0f : gamepadmag*gamepadmag;
        gamepadx = gamepaddirx * gamepadclippedmag;
        gamepady = gamepaddiry * gamepadclippedmag;
    }
    else
    {
        gamepadx = 0.0f;
        gamepady = 0.0f;
    }
    
    return vec3(gamepadx, 0.0f, gamepady);
}

//--------------------------------------

float orbit_camera_update_azimuth(
    const float azimuth, 
    const vec3 gamepadstick_right,
    const bool desired_strafe,
    const float dt)
{
    vec3 gamepadaxis = desired_strafe ? vec3() : gamepadstick_right;
    return azimuth + 2.0f * dt * -gamepadaxis.x;
}

float orbit_camera_update_altitude(
    const float altitude, 
    const vec3 gamepadstick_right,
    const bool desired_strafe,
    const float dt)
{
    vec3 gamepadaxis = desired_strafe ? vec3() : gamepadstick_right;
    return clampf(altitude + 2.0f * dt * gamepadaxis.z, 0.0, 0.4f * PIf);
}

float orbit_camera_update_distance(
    const float distance, 
    const float dt)
{
    float gamepadzoom = 
        (IsGamepadButtonDown(GAMEPAD_PLAYER, GAMEPAD_BUTTON_LEFT_TRIGGER_1) || IsKeyDown(KEY_Q))  ? +1.0f :
        (IsGamepadButtonDown(GAMEPAD_PLAYER, GAMEPAD_BUTTON_RIGHT_TRIGGER_1) || IsKeyDown(KEY_E)) ? -1.0f : 0.0f;
        
    return clampf(distance +  10.0f * dt * gamepadzoom, 0.1f, 100.0f);
}

// Updates the camera using the orbit cam controls
void orbit_camera_update(
    Camera3D& cam, 
    float& camera_azimuth,
    float& camera_altitude,
    float& camera_distance,
    const vec3 target,
    const vec3 gamepadstick_right,
    const bool desired_strafe,
    const float dt)
{
    camera_azimuth = orbit_camera_update_azimuth(camera_azimuth, gamepadstick_right, desired_strafe, dt);
    camera_altitude = orbit_camera_update_altitude(camera_altitude, gamepadstick_right, desired_strafe, dt);
    camera_distance = orbit_camera_update_distance(camera_distance, dt);
    
    quat rotation_azimuth = quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0));
    vec3 position = quat_mul_vec3(rotation_azimuth, vec3(0, 0, camera_distance));
    vec3 axis = normalize(cross(position, vec3(0, 1, 0)));
    
    quat rotation_altitude = quat_from_angle_axis(camera_altitude, axis);
    
    vec3 eye = target + quat_mul_vec3(rotation_altitude, position);

    cam.target = Vector3{ target.x, target.y, target.z };
    cam.position = Vector3{ eye.x, eye.y, eye.z };
}

//--------------------------------------

bool desired_strafe_update()
{
    return IsGamepadButtonDown(GAMEPAD_PLAYER, GAMEPAD_BUTTON_LEFT_TRIGGER_2) > 0.5f || IsKeyDown(KEY_LEFT_CONTROL);
}

void desired_gait_update(
    float& desired_gait, 
    float& desired_gait_velocity,
    const float dt,
    const float gait_change_halflife = 0.1f)
{
    simple_spring_damper_exact(
        desired_gait, 
        desired_gait_velocity,
        (IsGamepadButtonDown(GAMEPAD_PLAYER, GAMEPAD_BUTTON_RIGHT_FACE_DOWN) || IsKeyDown(KEY_LEFT_SHIFT)) ? 1.0f : 0.0f,
        gait_change_halflife,
        dt);
}

vec3 desired_velocity_update(
    const vec3 gamepadstick_left,
    const float camera_azimuth,
    const quat simulation_rotation,
    const float fwrd_speed,
    const float side_speed,
    const float back_speed)
{
    // Find stick position in world space by rotating using camera azimuth
    vec3 global_stick_direction = quat_mul_vec3(
        quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0)), gamepadstick_left);
    
    // Find stick position local to current facing direction
    vec3 local_stick_direction = quat_inv_mul_vec3(
        simulation_rotation, global_stick_direction);
    
    // Scale stick by forward, sideways and backwards speeds
    vec3 local_desired_velocity = local_stick_direction.z > 0.0 ?
        vec3(side_speed, 0.0f, fwrd_speed) * local_stick_direction :
        vec3(side_speed, 0.0f, back_speed) * local_stick_direction;
    
    // Re-orientate into the world space
    return quat_mul_vec3(simulation_rotation, local_desired_velocity);
}

quat desired_rotation_update(
    const quat desired_rotation,
    const vec3 gamepadstick_left,
    const vec3 gamepadstick_right,
    const float camera_azimuth,
    const bool desired_strafe,
    const vec3 desired_velocity)
{
    quat desired_rotation_curr = desired_rotation;
    
    // If strafe is active then desired direction is coming from right
    // stick as long as that stick is being used, otherwise we assume
    // forward facing
    if (desired_strafe)
    {
        vec3 desired_direction = quat_mul_vec3(quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0)), vec3(0, 0, -1));

        if (length(gamepadstick_right) > 0.01f)
        {
            desired_direction = quat_mul_vec3(quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0)), normalize(gamepadstick_right));
        }
        
        return quat_from_angle_axis(atan2f(desired_direction.x, desired_direction.z), vec3(0, 1, 0));            
    }
    
    // Travel never rewrites heading. Without an explicit right-stick heading
    // command, preserve the independently commanded orientation.
    (void)gamepadstick_left;
    (void)desired_velocity;
    return desired_rotation_curr;
}

//--------------------------------------

// Moving the root is a little bit difficult when we have the
// inertializer set up in the way we do. Essentially we need
// to also make sure to adjust all of the locations where 
// we are transforming the data to and from as well as the 
// offsets being blended out
void inertialize_root_adjust(
    vec3& offset_position,
    vec3& transition_src_position,
    quat& transition_src_rotation,
    vec3& transition_dst_position,
    quat& transition_dst_rotation,
    vec3& position,
    quat& rotation,
    const vec3 input_position,
    const quat input_rotation)
{
    // Find the position difference and add it to the state and transition location
    vec3 position_difference = input_position - position;
    position = position_difference + position;
    transition_dst_position = position_difference + transition_dst_position;
    
    // Find the point at which we want to now transition from in the src data
    transition_src_position = transition_src_position + quat_mul_vec3(transition_src_rotation,
        quat_inv_mul_vec3(transition_dst_rotation, position - offset_position - transition_dst_position));
    transition_dst_position = position;
    offset_position = vec3();
    
    // Find the rotation difference. We need to normalize here or some error can accumulate 
    // over time during adjustment.
    quat rotation_difference = quat_normalize(quat_mul_inv(input_rotation, rotation));
    
    // Apply the rotation difference to the current rotation and transition location
    rotation = quat_mul(rotation_difference, rotation);
    transition_dst_rotation = quat_mul(rotation_difference, transition_dst_rotation);
}

void inertialize_pose_reset(
    slice1d<vec3> bone_offset_positions,
    slice1d<vec3> bone_offset_velocities,
    slice1d<quat> bone_offset_rotations,
    slice1d<vec3> bone_offset_angular_velocities,
    vec3& transition_src_position,
    quat& transition_src_rotation,
    vec3& transition_dst_position,
    quat& transition_dst_rotation,
    const vec3 root_position,
    const quat root_rotation)
{
    bone_offset_positions.zero();
    bone_offset_velocities.zero();
    bone_offset_rotations.set(quat());
    bone_offset_angular_velocities.zero();
    
    transition_src_position = root_position;
    transition_src_rotation = root_rotation;
    transition_dst_position = vec3();
    transition_dst_rotation = quat();
}

// This function transitions the inertializer for 
// the full character. It takes as input the current 
// offsets, as well as the root transition locations,
// current root state, and the full pose information 
// for the pose being transitioned from (src) as well 
// as the pose being transitioned to (dst) in their
// own animation spaces.
void inertialize_pose_transition(
    slice1d<vec3> bone_offset_positions,
    slice1d<vec3> bone_offset_velocities,
    slice1d<quat> bone_offset_rotations,
    slice1d<vec3> bone_offset_angular_velocities,
    vec3& transition_src_position,
    quat& transition_src_rotation,
    vec3& transition_dst_position,
    quat& transition_dst_rotation,
    const vec3 root_position,
    const vec3 root_velocity,
    const quat root_rotation,
    const vec3 root_angular_velocity,
    const slice1d<vec3> bone_src_positions,
    const slice1d<vec3> bone_src_velocities,
    const slice1d<quat> bone_src_rotations,
    const slice1d<vec3> bone_src_angular_velocities,
    const slice1d<vec3> bone_dst_positions,
    const slice1d<vec3> bone_dst_velocities,
    const slice1d<quat> bone_dst_rotations,
    const slice1d<vec3> bone_dst_angular_velocities)
{
    // First we record the root position and rotation
    // in the animation data for the source and destination
    // animation
    transition_dst_position = root_position;
    transition_dst_rotation = root_rotation;
    transition_src_position = bone_dst_positions(0);
    transition_src_rotation = bone_dst_rotations(0);
    
    // We then find the velocities so we can transition the 
    // root inertiaizers
    vec3 world_space_dst_velocity = quat_mul_vec3(transition_dst_rotation, 
        quat_inv_mul_vec3(transition_src_rotation, bone_dst_velocities(0)));
    
    vec3 world_space_dst_angular_velocity = quat_mul_vec3(transition_dst_rotation, 
        quat_inv_mul_vec3(transition_src_rotation, bone_dst_angular_velocities(0)));
    
    // Transition inertializers recording the offsets for 
    // the root joint
    inertialize_transition(
        bone_offset_positions(0),
        bone_offset_velocities(0),
        root_position,
        root_velocity,
        root_position,
        world_space_dst_velocity);
        
    inertialize_transition(
        bone_offset_rotations(0),
        bone_offset_angular_velocities(0),
        root_rotation,
        root_angular_velocity,
        root_rotation,
        world_space_dst_angular_velocity);
    
    // Transition all the inertializers for each other bone
    for (int i = 1; i < bone_offset_positions.size; i++)
    {
        inertialize_transition(
            bone_offset_positions(i),
            bone_offset_velocities(i),
            bone_src_positions(i),
            bone_src_velocities(i),
            bone_dst_positions(i),
            bone_dst_velocities(i));
            
        inertialize_transition(
            bone_offset_rotations(i),
            bone_offset_angular_velocities(i),
            bone_src_rotations(i),
            bone_src_angular_velocities(i),
            bone_dst_rotations(i),
            bone_dst_angular_velocities(i));
    }
}

// This function updates the inertializer states. Here 
// it outputs the smoothed animation (input plus offset) 
// as well as updating the offsets themselves. It takes 
// as input the current playing animation as well as the 
// root transition locations, a halflife, and a dt
void inertialize_pose_update(
    slice1d<vec3> bone_positions,
    slice1d<vec3> bone_velocities,
    slice1d<quat> bone_rotations,
    slice1d<vec3> bone_angular_velocities,
    slice1d<vec3> bone_offset_positions,
    slice1d<vec3> bone_offset_velocities,
    slice1d<quat> bone_offset_rotations,
    slice1d<vec3> bone_offset_angular_velocities,
    const slice1d<vec3> bone_input_positions,
    const slice1d<vec3> bone_input_velocities,
    const slice1d<quat> bone_input_rotations,
    const slice1d<vec3> bone_input_angular_velocities,
    const vec3 transition_src_position,
    const quat transition_src_rotation,
    const vec3 transition_dst_position,
    const quat transition_dst_rotation,
    const float halflife,
    const float dt)
{
    // First we find the next root position, velocity, rotation
    // and rotational velocity in the world space by transforming 
    // the input animation from it's animation space into the 
    // space of the currently playing animation.
    vec3 world_space_position = quat_mul_vec3(transition_dst_rotation, 
        quat_inv_mul_vec3(transition_src_rotation, 
            bone_input_positions(0) - transition_src_position)) + transition_dst_position;
    
    vec3 world_space_velocity = quat_mul_vec3(transition_dst_rotation, 
        quat_inv_mul_vec3(transition_src_rotation, bone_input_velocities(0)));
    
    // Normalize here because quat inv mul can sometimes produce 
    // unstable returns when the two rotations are very close.
    quat world_space_rotation = quat_normalize(quat_mul(transition_dst_rotation, 
        quat_inv_mul(transition_src_rotation, bone_input_rotations(0))));
    
    vec3 world_space_angular_velocity = quat_mul_vec3(transition_dst_rotation, 
        quat_inv_mul_vec3(transition_src_rotation, bone_input_angular_velocities(0)));
    
    // Then we update these two inertializers with these new world space inputs
    inertialize_update(
        bone_positions(0),
        bone_velocities(0),
        bone_offset_positions(0),
        bone_offset_velocities(0),
        world_space_position,
        world_space_velocity,
        halflife,
        dt);
        
    inertialize_update(
        bone_rotations(0),
        bone_angular_velocities(0),
        bone_offset_rotations(0),
        bone_offset_angular_velocities(0),
        world_space_rotation,
        world_space_angular_velocity,
        halflife,
        dt);        
    
    // Then we update the inertializers for the rest of the bones
    for (int i = 1; i < bone_positions.size; i++)
    {
        inertialize_update(
            bone_positions(i),
            bone_velocities(i),
            bone_offset_positions(i),
            bone_offset_velocities(i),
            bone_input_positions(i),
            bone_input_velocities(i),
            halflife,
            dt);
            
        inertialize_update(
            bone_rotations(i),
            bone_angular_velocities(i),
            bone_offset_rotations(i),
            bone_offset_angular_velocities(i),
            bone_input_rotations(i),
            bone_input_angular_velocities(i),
            halflife,
            dt);
    }
}

//--------------------------------------

// Copy a part of a feature vector from the 
// matching database into the query feature vector
void query_copy_denormalized_feature(
    slice1d<float> query, 
    int& offset, 
    const int size, 
    const slice1d<float> features,
    const slice1d<float> features_offset,
    const slice1d<float> features_scale)
{
    for (int i = 0; i < size; i++)
    {
        query(offset + i) = features(offset + i) * features_scale(offset + i) + features_offset(offset + i);
    }
    
    offset += size;
}

// Compute the query feature vector for the current 
// trajectory controlled by the gamepad.
void query_compute_trajectory_position_feature(
    slice1d<float> query, 
    int& offset, 
    const vec3 root_position, 
    const quat root_rotation, 
    const slice1d<vec3> trajectory_positions)
{
    vec3 traj0 = quat_inv_mul_vec3(root_rotation, trajectory_positions(1) - root_position);
    vec3 traj1 = quat_inv_mul_vec3(root_rotation, trajectory_positions(2) - root_position);
    vec3 traj2 = quat_inv_mul_vec3(root_rotation, trajectory_positions(3) - root_position);
    
    query(offset + 0) = traj0.x;
    query(offset + 1) = traj0.z;
    query(offset + 2) = traj1.x;
    query(offset + 3) = traj1.z;
    query(offset + 4) = traj2.x;
    query(offset + 5) = traj2.z;
    
    offset += 6;
}

// Same but for the trajectory direction
void query_compute_trajectory_direction_feature(
    slice1d<float> query, 
    int& offset, 
    const quat root_rotation, 
    const slice1d<quat> trajectory_rotations)
{
    vec3 traj0 = quat_inv_mul_vec3(root_rotation, quat_mul_vec3(trajectory_rotations(1), vec3(0, 0, 1)));
    vec3 traj1 = quat_inv_mul_vec3(root_rotation, quat_mul_vec3(trajectory_rotations(2), vec3(0, 0, 1)));
    vec3 traj2 = quat_inv_mul_vec3(root_rotation, quat_mul_vec3(trajectory_rotations(3), vec3(0, 0, 1)));
    
    query(offset + 0) = traj0.x;
    query(offset + 1) = traj0.z;
    query(offset + 2) = traj1.x;
    query(offset + 3) = traj1.z;
    query(offset + 4) = traj2.x;
    query(offset + 5) = traj2.z;
    
    offset += 6;
}

//--------------------------------------

// Taken from https://theorangeduck.com/page/spring-roll-call#controllers
void simulation_positions_update(
    vec3& position, 
    vec3& velocity, 
    vec3& acceleration, 
    const vec3 desired_velocity, 
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f; 
    vec3 j0 = velocity - desired_velocity;
    vec3 j1 = acceleration + j0*y;
    float eydt = fast_negexpf(y*dt);
    
    vec3 position_prev = position;

    position = eydt*(((-j1)/(y*y)) + ((-j0 - j1*dt)/y)) + 
        (j1/(y*y)) + j0/y + desired_velocity * dt + position_prev;
    velocity = eydt*(j0 + j1*dt) + desired_velocity;
    acceleration = eydt*(acceleration - j1*y*dt);
    
}

void simulation_rotations_update(
    quat& rotation, 
    vec3& angular_velocity, 
    const quat desired_rotation, 
    const float halflife, 
    const float dt)
{
    simple_spring_damper_exact(
        rotation, 
        angular_velocity, 
        desired_rotation, 
        halflife, dt);
}

// Predict what the desired velocity will be in the 
// future. Here we need to use the future trajectory 
// rotation as well as predicted future camera 
// position to find an accurate desired velocity in 
// the world space
void trajectory_desired_velocities_predict(
  slice1d<vec3> desired_velocities,
  const slice1d<quat> trajectory_rotations,
  const vec3 desired_velocity,
  const float camera_azimuth,
  const vec3 gamepadstick_left,
  const vec3 gamepadstick_right,
  const bool desired_strafe,
  const float fwrd_speed,
  const float side_speed,
  const float back_speed,
  const float dt)
{
    desired_velocities(0) = desired_velocity;
    
    for (int i = 1; i < desired_velocities.size; i++)
    {
        desired_velocities(i) = desired_velocity_update(
            gamepadstick_left,
            orbit_camera_update_azimuth(
                camera_azimuth, gamepadstick_right, desired_strafe, i * dt),
            trajectory_rotations(i),
            fwrd_speed,
            side_speed,
            back_speed);
    }
}

void trajectory_positions_predict(
    slice1d<vec3> positions, 
    slice1d<vec3> velocities, 
    slice1d<vec3> accelerations, 
    const vec3 position, 
    const vec3 velocity, 
    const vec3 acceleration, 
    const slice1d<vec3> desired_velocities, 
    const float halflife,
    const float dt)
{
    positions(0) = position;
    velocities(0) = velocity;
    accelerations(0) = acceleration;
    
    for (int i = 1; i < positions.size; i++)
    {
        positions(i) = positions(i-1);
        velocities(i) = velocities(i-1);
        accelerations(i) = accelerations(i-1);
        
        simulation_positions_update(
            positions(i), 
            velocities(i), 
            accelerations(i), 
            desired_velocities(i), 
            halflife, 
            dt);
    }
}

// Predict desired rotations given the estimated future 
// camera rotation and other parameters
void trajectory_desired_rotations_predict(
  slice1d<quat> desired_rotations,
  const slice1d<vec3> desired_velocities,
  const quat desired_rotation,
  const float camera_azimuth,
  const vec3 gamepadstick_left,
  const vec3 gamepadstick_right,
  const bool desired_strafe,
  const float dt)
{
    desired_rotations(0) = desired_rotation;
    
    for (int i = 1; i < desired_rotations.size; i++)
    {
        desired_rotations(i) = desired_rotation_update(
            desired_rotations(i-1),
            gamepadstick_left,
            gamepadstick_right,
            orbit_camera_update_azimuth(
                camera_azimuth, gamepadstick_right, desired_strafe, i * dt),
            desired_strafe,
            desired_velocities(i));
    }
}

void trajectory_rotations_predict(
    slice1d<quat> rotations, 
    slice1d<vec3> angular_velocities, 
    const quat rotation, 
    const vec3 angular_velocity, 
    const slice1d<quat> desired_rotations, 
    const float halflife,
    const float dt)
{
    rotations.set(rotation);
    angular_velocities.set(angular_velocity);
    
    for (int i = 1; i < rotations.size; i++)
    {
        simulation_rotations_update(
            rotations(i), 
            angular_velocities(i), 
            desired_rotations(i), 
            halflife, 
            i * dt);
    }
}

//--------------------------------------

void contact_reset(
    bool& contact_state,
    bool& contact_lock,
    vec3& contact_position,
    vec3& contact_velocity,
    vec3& contact_point,
    vec3& contact_target,
    vec3& contact_offset_position,
    vec3& contact_offset_velocity,
    const vec3 input_contact_position,
    const vec3 input_contact_velocity,
    const bool input_contact_state)
{
    (void)input_contact_state;

    contact_state = false;
    contact_lock = false;
    contact_position = input_contact_position;
    contact_velocity = input_contact_velocity;
    contact_point = input_contact_position;
    contact_target = input_contact_position;
    contact_offset_position = vec3();
    contact_offset_velocity = vec3();
}

void contact_update(
    bool& contact_state,
    bool& contact_lock,
    vec3& contact_position,
    vec3& contact_velocity,
    vec3& contact_point,
    vec3& contact_target,
    vec3& contact_offset_position,
    vec3& contact_offset_velocity,
    const vec3 input_contact_position,
    const bool input_contact_state,
    const float unlock_radius,
    const float foot_height,
    const float halflife,
    const float dt,
    const float eps=1e-8)
{
    // First compute the input contact position velocity via finite difference
    vec3 input_contact_velocity = 
        (input_contact_position - contact_target) / (dt + eps);    
    contact_target = input_contact_position;
    
    // Update the inertializer to tick forward in time
    inertialize_update(
        contact_position,
        contact_velocity,
        contact_offset_position,
        contact_offset_velocity,
        // If locked we feed the contact point and zero velocity, 
        // otherwise we feed the input from the animation
        contact_lock ? contact_point : input_contact_position,
        contact_lock ?        vec3() : input_contact_velocity,
        halflife,
        dt);
    
    // If the contact point is too far from the current input position 
    // then we need to unlock the contact
    bool unlock_contact = contact_lock && (
        length(contact_point - input_contact_position) > unlock_radius);
    
    // If the contact was previously inactive but is now active we 
    // need to transition to the locked contact state
    if (!contact_state && input_contact_state)
    {
        // Contact point is given by the current position of 
        // the foot projected onto the ground plus foot height
        contact_lock = true;
        contact_point = contact_position;
        contact_point.y = foot_height;
        
        inertialize_transition(
            contact_offset_position,
            contact_offset_velocity,
            input_contact_position,
            input_contact_velocity,
            contact_point,
            vec3());
    }
    
    // Otherwise if we need to unlock or we were previously in 
    // contact but are no longer we transition to just taking 
    // the input position as-is
    else if ((contact_lock && contact_state && !input_contact_state) 
         || unlock_contact)
    {
        contact_lock = false;
        
        inertialize_transition(
            contact_offset_position,
            contact_offset_velocity,
            contact_point,
            vec3(),
            input_contact_position,
            input_contact_velocity);
    }
    
    // Update contact state
    contact_state = input_contact_state;
}

//--------------------------------------

// Rotate a joint to look toward some 
// given target position
void ik_look_at(
    quat& bone_rotation,
    const quat global_parent_rotation,
    const quat global_rotation,
    const vec3 global_position,
    const vec3 child_position,
    const vec3 target_position,
    const float eps = 1e-5f)
{
    vec3 curr_dir = normalize(child_position - global_position);
    vec3 targ_dir = normalize(target_position - global_position);

    if (fabs(1.0f - dot(curr_dir, targ_dir) > eps))
    {
        bone_rotation = quat_inv_mul(global_parent_rotation, 
            quat_mul(quat_between(curr_dir, targ_dir), global_rotation));
    }
}

// Basic two-joint IK in the style of https://theorangeduck.com/page/simple-two-joint
// Here I add a basic "forward vector" which acts like a kind of pole-vetor
// to control the bending direction
void ik_two_bone(
    quat& bone_root_lr, 
    quat& bone_mid_lr,
    const vec3 bone_root, 
    const vec3 bone_mid, 
    const vec3 bone_end, 
    const vec3 target, 
    const vec3 fwd,
    const quat bone_root_gr, 
    const quat bone_mid_gr,
    const quat bone_par_gr,
    const float max_length_buffer) {
    
    float max_extension = 
        length(bone_root - bone_mid) + 
        length(bone_mid - bone_end) - 
        max_length_buffer;
    
    vec3 target_clamp = target;
    if (length(target - bone_root) > max_extension)
    {
        target_clamp = bone_root + max_extension * normalize(target - bone_root);
    }
    
    vec3 axis_dwn = normalize(bone_end - bone_root);
    vec3 axis_rot = normalize(cross(axis_dwn, fwd));

    vec3 a = bone_root;
    vec3 b = bone_mid;
    vec3 c = bone_end;
    vec3 t = target_clamp;
    
    float lab = length(b - a);
    float lcb = length(b - c);
    float lat = length(t - a);

    float ac_ab_0 = acosf(clampf(dot(normalize(c - a), normalize(b - a)), -1.0f, 1.0f));
    float ba_bc_0 = acosf(clampf(dot(normalize(a - b), normalize(c - b)), -1.0f, 1.0f));

    float ac_ab_1 = acosf(clampf((lab * lab + lat * lat - lcb * lcb) / (2.0f * lab * lat), -1.0f, 1.0f));
    float ba_bc_1 = acosf(clampf((lab * lab + lcb * lcb - lat * lat) / (2.0f * lab * lcb), -1.0f, 1.0f));

    quat r0 = quat_from_angle_axis(ac_ab_1 - ac_ab_0, axis_rot);
    quat r1 = quat_from_angle_axis(ba_bc_1 - ba_bc_0, axis_rot);

    vec3 c_a = normalize(bone_end - bone_root);
    vec3 t_a = normalize(target_clamp - bone_root);

    quat r2 = quat_from_angle_axis(
        acosf(clampf(dot(c_a, t_a), -1.0f, 1.0f)),
        normalize(cross(c_a, t_a)));
    
    bone_root_lr = quat_inv_mul(bone_par_gr, quat_mul(r2, quat_mul(r0, bone_root_gr)));
    bone_mid_lr = quat_inv_mul(bone_root_gr, quat_mul(r1, bone_mid_gr));
}

//--------------------------------------

void draw_axis(const vec3 pos, const quat rot, const float scale = 1.0f)
{
    vec3 axis0 = pos + quat_mul_vec3(rot, scale * vec3(1.0f, 0.0f, 0.0f));
    vec3 axis1 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 1.0f, 0.0f));
    vec3 axis2 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 0.0f, 1.0f));
    
    DrawLine3D(to_Vector3(pos), to_Vector3(axis0), RED);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis1), GREEN);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis2), BLUE);
}

void draw_features(const slice1d<float> features, const vec3 pos, const quat rot, const Color color)
{
    vec3 lfoot_pos = quat_mul_vec3(rot, vec3(features( 0), features( 1), features( 2))) + pos;
    vec3 rfoot_pos = quat_mul_vec3(rot, vec3(features( 3), features( 4), features( 5))) + pos;
    vec3 lfoot_vel = quat_mul_vec3(rot, vec3(features( 6), features( 7), features( 8)));
    vec3 rfoot_vel = quat_mul_vec3(rot, vec3(features( 9), features(10), features(11)));
    //vec3 hip_vel   = quat_mul_vec3(rot, vec3(features(12), features(13), features(14)));
    vec3 traj0_pos = quat_mul_vec3(rot, vec3(features(15),         0.0f, features(16))) + pos;
    vec3 traj1_pos = quat_mul_vec3(rot, vec3(features(17),         0.0f, features(18))) + pos;
    vec3 traj2_pos = quat_mul_vec3(rot, vec3(features(19),         0.0f, features(20))) + pos;
    vec3 traj0_dir = quat_mul_vec3(rot, vec3(features(21),         0.0f, features(22)));
    vec3 traj1_dir = quat_mul_vec3(rot, vec3(features(23),         0.0f, features(24)));
    vec3 traj2_dir = quat_mul_vec3(rot, vec3(features(25),         0.0f, features(26)));
    
    DrawSphereWires(to_Vector3(lfoot_pos), 0.05f, 4, 10, color);
    DrawSphereWires(to_Vector3(rfoot_pos), 0.05f, 4, 10, color);
    DrawSphereWires(to_Vector3(traj0_pos), 0.05f, 4, 10, color);
    DrawSphereWires(to_Vector3(traj1_pos), 0.05f, 4, 10, color);
    DrawSphereWires(to_Vector3(traj2_pos), 0.05f, 4, 10, color);
    
    DrawLine3D(to_Vector3(lfoot_pos), to_Vector3(lfoot_pos + 0.1f * lfoot_vel), color);
    DrawLine3D(to_Vector3(rfoot_pos), to_Vector3(rfoot_pos + 0.1f * rfoot_vel), color);
    
    DrawLine3D(to_Vector3(traj0_pos), to_Vector3(traj0_pos + 0.25f * traj0_dir), color);
    DrawLine3D(to_Vector3(traj1_pos), to_Vector3(traj1_pos + 0.25f * traj1_dir), color);
    DrawLine3D(to_Vector3(traj2_pos), to_Vector3(traj2_pos + 0.25f * traj2_dir), color); 
}

void draw_trajectory(
    const slice1d<vec3> trajectory_positions, 
    const slice1d<quat> trajectory_rotations, 
    const Color color)
{
    for (int i = 1; i < trajectory_positions.size; i++)
    {
        DrawSphereWires(to_Vector3(trajectory_positions(i)), 0.05f, 4, 10, color);
        DrawLine3D(to_Vector3(trajectory_positions(i)), to_Vector3(
            trajectory_positions(i) + 0.6f * quat_mul_vec3(trajectory_rotations(i), vec3(0, 0, 1.0f))), color);
        DrawLine3D(to_Vector3(trajectory_positions(i-1)), to_Vector3(trajectory_positions(i)), color);
    }
}

quat adjust_character_rotation(
    const quat character_rotation,
    const quat simulation_rotation,
    const float halflife,
    const float dt)
{
    // Find the difference in rotation (from character to simulation).
    // Here `quat_abs` forces the quaternion to take the shortest 
    // path and normalization is required as sometimes taking 
    // the difference between two very similar rotations can 
    // introduce numerical instability
    quat difference_rotation = quat_abs(quat_normalize(
        quat_mul_inv(simulation_rotation, character_rotation)));
    
    // Damp that difference using the given halflife and dt
    quat adjustment_rotation = damp_adjustment_exact(
        difference_rotation,
        halflife,
        dt);
    
    // Apply the damped adjustment to the character
    return quat_mul(adjustment_rotation, character_rotation);
}

quat adjust_character_rotation_by_velocity(
    const quat character_rotation,
    const vec3 character_angular_velocity,
    const quat simulation_rotation,
    const float max_adjustment_ratio,
    const float halflife,
    const float dt)
{
    // Find and damp the desired rotational adjustment
    quat adjustment_rotation = damp_adjustment_exact(
        quat_abs(quat_normalize(quat_mul_inv(
            simulation_rotation, character_rotation))),
        halflife,
        dt);
    
    // If the length of the adjustment is greater than the angular velocity 
    // multiplied by the ratio then we need to clamp this adjustment
    float max_length = max_adjustment_ratio *
        length(character_angular_velocity) * dt;
    
    if (length(quat_to_scaled_angle_axis(adjustment_rotation)) > max_length)
    {
        // To clamp can convert to scaled angle axis, rescale, and convert back
        adjustment_rotation = quat_from_scaled_angle_axis(max_length * 
            normalize(quat_to_scaled_angle_axis(adjustment_rotation)));
    }
    
    // Apply the adjustment
    return quat_mul(adjustment_rotation, character_rotation);
}

//--------------------------------------

quat clamp_character_rotation(
    const quat character_rotation,
    const quat simulation_rotation,
    const float max_angle)
{
    // If the angle between the character rotation and simulation 
    // rotation exceeds the threshold we need to clamp it back
    if (quat_angle_between(character_rotation, simulation_rotation) > max_angle)
    {
        // First, find the rotational difference between the two
        quat diff = quat_abs(quat_mul_inv(
            character_rotation, simulation_rotation));
        
        // We can then decompose it into angle and axis
        float diff_angle; vec3 diff_axis;
        quat_to_angle_axis(diff, diff_angle, diff_axis);
        
        // We then clamp the angle to within our bounds
        diff_angle = clampf(diff_angle, -max_angle, max_angle);
        
        // And apply back the clamped rotation
        return quat_mul(
          quat_from_angle_axis(diff_angle, diff_axis), simulation_rotation);
    }
    else
    {
        return character_rotation;
    }
}

//--------------------------------------

void update_callback(void* args)
{
    ((std::function<void()>*)args)->operator()();
}

int main(void)
{
    const int screen_width = 1280;
    const int screen_height = 720;

    const char* terrain_directory = getenv("G1_TERRAIN_DIR");
    if (terrain_directory == NULL)
    {
        terrain_directory = "./resources/g1_terrain";
    }
    char artifact_error[512] = {};

    float feature_weight_foot_position = 0.75f;
    float feature_weight_foot_velocity = 1.0f;
    float feature_weight_hip_velocity = 1.0f;
    float feature_weight_trajectory_positions = 1.0f;
    float feature_weight_trajectory_directions = 1.5f;
    float parsed_terrain_weight = 4.0f;
    g1_test_config test_config;
    G1TestHeadingOverride test_heading;
    if (!g1_parse_terrain_weight(
            parsed_terrain_weight,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        !g1_parse_test_config(
            test_config,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        !g1_test_heading_override_parse(
            test_heading,
            getenv("MM_TEST_HEADING"),
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 terrain option error: %s\n", artifact_error);
        return 2;
    }
    if (test_heading.active && test_config.mode != G1_TestRoute) {
        fprintf(
            stderr,
            "G1 terrain option error: MM_TEST_HEADING is valid only for "
            "route mode\n");
        return 2;
    }
#if defined(PLATFORM_WEB)
    // Deterministic controller tests require a bounded desktop update loop.
    if (test_config.mode != G1_TestLive || test_config.frame_limit > 0) {
        fprintf(stderr,
            "G1 terrain option error: bounded MM_TEST_MODE/MM_TEST_FRAMES "
            "runs are desktop-only\n");
        return 2;
    }
#if !defined(MM_DISCRETE)
    // Deterministic CSV gate evidence remains a desktop-only contract.
    if (getenv("MM_LOG") != NULL) {
        fprintf(stderr,
            "G1 runtime log error: deterministic CSV MM_LOG is "
            "desktop-only\n");
        return 2;
    }
#endif
#endif
    if (test_config.mode == G1_TestFlat ||
        test_config.mode == G1_TestSequential) {
        parsed_terrain_weight = 0.0f;
    } else if (test_config.mode == G1_TestTerrain &&
               getenv("MM_TERRAIN_WEIGHT") == NULL) {
        parsed_terrain_weight = 4.0f;
    }
#ifdef MM_DISCRETE
    if (test_config.mode != G1_TestLive) {
        fprintf(stderr,
            "G1 terrain option error: MM_TEST_MODE is unavailable in "
            "MM_DISCRETE builds\n");
        return 2;
    }
#endif

    float requested_terrain_weight = parsed_terrain_weight;
    float effective_terrain_weight = requested_terrain_weight;

    motion_pack_manifest motion_manifest;
    if (!motion_manifest_load_and_verify(
            motion_manifest,
            terrain_directory,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 motion manifest error: %s\n", artifact_error);
        return 2;
    }

    std::string database_path;
    std::string feature_path;
    std::string support_path;
    std::string motion_index_path;
    if (!scene_join(
            database_path,
            terrain_directory,
            motion_manifest.database.path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        !scene_join(
            feature_path,
            terrain_directory,
            motion_manifest.terrain_features.path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        !scene_join(
            support_path,
            terrain_directory,
            motion_manifest.terrain_support.path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        !scene_join(
            motion_index_path,
            terrain_directory,
            motion_manifest.motion_index.path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 motion path error: %s\n", artifact_error);
        return 2;
    }

    terrain_feature_set terrain_rows;
    if (!terrain_features_load(
            terrain_rows,
            feature_path.c_str(),
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 terrain feature error: %s\n", artifact_error);
        return 2;
    }

    database db;
    database_load(db, database_path.c_str());
    if (!g1_database_validate(
            db, artifact_error, static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 database error: %s\n", artifact_error);
        return 2;
    }
    if (terrain_rows.values.rows != db.nframes() ||
        terrain_rows.values.cols != DATABASE_G1_TERRAIN_FEATURES)
    {
        fprintf(
            stderr,
            "G1 database/G1TF shape error: database=%d terrain=%dx%d "
            "expected_columns=12\n",
            db.nframes(),
            terrain_rows.values.rows,
            terrain_rows.values.cols);
        return 2;
    }
    db.terrain_features = terrain_rows.values;
    if (!g1_skeleton_validate(
            db, artifact_error, static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 skeleton error: %s\n", artifact_error);
        return 2;
    }
    if (!g1_leg_configs_validate(
            db, artifact_error, static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 IK geometry error: %s\n", artifact_error);
        return 2;
    }
    database_build_matching_features(
        db,
        feature_weight_foot_position,
        feature_weight_foot_velocity,
        feature_weight_hip_velocity,
        feature_weight_trajectory_positions,
        feature_weight_trajectory_directions,
        G1_LeftAnkle,
        G1_RightAnkle,
        G1_Hips,
        effective_terrain_weight);
    if (!g1_matching_features_validate(
            db, artifact_error, static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 feature error: %s\n", artifact_error);
        return 2;
    }

    motion_index_runtime motion_index;
    if (!motion_index_load(
            motion_index,
            motion_index_path.c_str(),
            db.range_starts.data,
            db.range_stops.data,
            static_cast<size_t>(db.nranges()),
            BOUND_SM_SIZE,
            BOUND_LR_SIZE,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 motion index error: %s\n", artifact_error);
        return 2;
    }

    if (!motion_manifest_validate_database(
            motion_manifest,
            db,
            motion_index,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(
            stderr,
            "G1 motion/database contract error: %s\n",
            artifact_error);
        return 2;
    }

    scene_catalog catalog;
    if (!scene_catalog_load(
            catalog,
            terrain_directory,
            motion_manifest,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 scene index error: %s\n", artifact_error);
        return 2;
    }
    const char* requested_scene = std::getenv("MM_TERRAIN_SCENE");
    if (requested_scene == NULL)
    {
        requested_scene = catalog.default_scene_id.c_str();
    }
    int active_scene_index = scene_catalog_find(catalog, requested_scene);
    if (active_scene_index < 0)
    {
        fprintf(
            stderr,
            "G1 scene selection error: unknown MM_TERRAIN_SCENE '%s'\n",
            requested_scene);
        return 2;
    }

    scene_pack active_scene;
    if (!scene_pack_load(
            active_scene,
            terrain_directory,
            motion_manifest,
            catalog,
            active_scene_index,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(
            stderr,
            "G1 scene error [%s]: %s\n",
            requested_scene,
            artifact_error);
        return 2;
    }

    int configured_route_index = -1;
    float configured_route_target_height = 0.0f;
    if (test_config.mode == G1_TestRoute) {
        const scene_route* configured_route = scene_route_find(
            active_scene.metadata, test_config.route);
        if (configured_route == NULL) {
            fprintf(
                stderr,
                "G1 route selection error: scene '%s' has no route '%s'\n",
                active_scene.metadata.id.c_str(),
                test_config.route);
            return 2;
        }
        configured_route_index = static_cast<int>(
            configured_route - active_scene.metadata.routes.data());
        configured_route_target_height = deterministic_route_target_height(
            *configured_route, active_scene.terrain);
        if (deterministic_route_motion_frames(*configured_route) < 0 ||
            !terrain_float_is_finite(configured_route_target_height))
        {
            fprintf(
                stderr,
                "G1 route selection error: scene '%s' route '%s' has invalid "
                "binary32 runtime metadata\n",
                active_scene.metadata.id.c_str(),
                test_config.route);
            return 2;
        }
    }

    terrain_support_set support_rows;
    if (!terrain_support_load(
            support_rows,
            support_path.c_str(),
            db.nframes(),
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 support error: %s\n", artifact_error);
        return 2;
    }

    g1_controller_state state;
    if (!g1_controller_state_reset(
            state,
            db,
            support_rows,
            active_scene,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 reset error: %s\n", artifact_error);
        return 2;
    }
    auto configure_route_cursor = [&](g1_controller_state& current)
    {
        current.route_index = configured_route_index;
        current.route_waypoint = configured_route_index >= 0 ? 1 : 0;
        current.route_frames = 0;
    };
    configure_route_cursor(state);
    motion_bank_state bank_state;
    motion_bank_state_reset(bank_state);
    bool coverage_safe_stop_latched = false;

    if (test_config.mode == G1_TestSequential) {
        const int sequential_frames =
            db.range_stops(0) - db.range_starts(0) - 1;
        if (test_config.frame_limit > sequential_frames) {
            fprintf(stderr,
                "G1 sequential test overrun: MM_TEST_FRAMES=%d exceeds "
                "the initial range capacity %d\n",
                test_config.frame_limit, sequential_frames);
            return 2;
        }
    }

    // Bounded deterministic runs exercise the complete controller and log
    // contract without paying for X11 or render-model startup. Live mode keeps
    // the classic visualizer unchanged.
    const bool rendering_enabled = test_config.mode == G1_TestLive;
    if (rendering_enabled)
    {
        // Open the graphics window only after every artifact and feature gate
        // has passed, so startup errors stay useful on headless systems.
        SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(
            screen_width,
            screen_height,
            "G1 terrain motion matching - Holden runtime");
        if (!IsWindowReady())
        {
            fprintf(stderr, "G1 terrain visualizer could not open a window\n");
            return 2;
        }
        SetTargetFPS(25);
    }

    int scene_generation = 0;
    int scene_reset_count = 1;
    int motion_pack_load_count = 1;
    int model_load_count = 0;
    int model_unload_count = 0;
    bool scene_switch_failed = false;
    bool controller_exit_requested = false;
    int controller_exit_code = 0;

    Model terrain_model = {};
    auto model_has_allocation = [](const Model& model)
    {
        return model.meshes != NULL || model.materials != NULL ||
               model.meshMaterial != NULL || model.bones != NULL ||
               model.bindPose != NULL;
    };
    bool terrain_model_allocated = false;
    if (!rendering_enabled)
    {
        // Preserve the existing one-live-scene ownership diagnostics while
        // representing the omitted render model with a zero-allocation token.
        ++model_load_count;
    }
    else
    {
        terrain_model = LoadModel(active_scene.mesh_path.c_str());
        terrain_model_allocated = model_has_allocation(terrain_model);
        if (terrain_model_allocated) ++model_load_count;
        if (!IsModelReady(terrain_model) || terrain_model.meshCount <= 0)
        {
            fprintf(
                stderr,
                "G1 terrain mesh failed to load: %s\n",
                active_scene.mesh_path.c_str());
            controller_exit_code = 2;
            controller_exit_requested = true;
        }
    }

    G1MeshRenderer g1_mesh_renderer = {};
    bool show_g1_mesh = true;
    bool show_g1_bones = true;
    if (rendering_enabled && !controller_exit_requested &&
        !::g1_mesh_renderer_load(
            g1_mesh_renderer,
            "resources/g1_mesh/g1_raylib.glb",
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        std::fprintf(stderr, "G1 mesh load error: %s\n", artifact_error);
        controller_exit_code = 2;
        controller_exit_requested = true;
    }
    if (rendering_enabled && !controller_exit_requested)
    {
        std::fprintf(
            stdout,
            "G1 mesh loaded: %d parts\n",
            g1_mesh_renderer.model.meshCount);
    }

    auto scene_loader = [&](scene_pack& candidate, int index,
                            char* error, int capacity)
    {
        return scene_pack_load(
            candidate,
            terrain_directory,
            motion_manifest,
            catalog,
            index,
            error,
            capacity);
    };
    auto model_loader = [&](Model& model, const char* path,
                            char* error, int capacity)
    {
        if (!rendering_enabled)
        {
            model = Model{};
            ++model_load_count;
            return scene_model_load_result{false, true};
        }
        model = LoadModel(path);
        const bool allocated = model_has_allocation(model);
        if (allocated) ++model_load_count;
        const bool ready = IsModelReady(model) && model.meshCount > 0;
        if (!ready)
        {
            scene_error(error, capacity, "%s: Raylib model is not ready", path);
        }
        return scene_model_load_result{allocated, ready};
    };
    auto model_unloader = [&](Model& model)
    {
        if (!rendering_enabled)
        {
            model = Model{};
            ++model_unload_count;
            return;
        }
        if (model_has_allocation(model))
        {
            UnloadModel(model);
            ++model_unload_count;
        }
        model = Model{};
    };
    int pending_scene_index = -1;
    bool pending_reset = false;
    const int scene_count = static_cast<int>(catalog.ids.size());
    
    // Camera

    Camera3D camera{};
    camera.position = Vector3{ 0.0f, 10.0f, 10.0f };
    camera.target = Vector3{ 0.0f, 0.0f, 0.0f };
    camera.up = Vector3{ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    // Character
    
    // G1: no character.bin skinned mesh — the skeleton is drawn directly from
    // bone transforms in the render loop, so mesh/shader loading is skipped.

    // Pose & Inertializer Data
    
    float inertialize_blending_halflife = 0.10f;
        
    // Trajectory & Gameplay Data

#ifdef MM_DISCRETE
    if (const char* e = getenv("MM_SEARCHT")) {
        state.search_time = atof(e);
        state.search_timer = state.search_time;
        state.force_search_timer = state.search_time;
    }
#endif
    float desired_velocity_change_threshold = 50.0;
    float desired_rotation_change_threshold = 50.0;
    
    float simulation_velocity_halflife = 0.27f;
    float simulation_rotation_halflife = 0.27f;
    
    // All speeds in m/s
    // G1: takara walk DB only contains ~0.4-0.5 m/s of motion. Commanding
    // 4 m/s made the sim target outrun the feet (sliding), thrash the match
    // (leg twitch) and spin the root. Match speeds to what the data provides.
    float simulation_run_fwrd_speed = 0.9f;
    float simulation_run_side_speed = 0.6f;
    float simulation_run_back_speed = 0.6f;
    
    float simulation_walk_fwrd_speed = 0.5f;
    float simulation_walk_side_speed = 0.4f;
    float simulation_walk_back_speed = 0.4f;
    
    // Synchronization
    
    bool synchronization_enabled = false;
    float synchronization_data_factor = 1.0f;
    
    // Adjustment
    
    bool adjustment_enabled = true;
    bool adjustment_by_velocity_enabled = true;
    float adjustment_position_halflife = 0.1f;
    float adjustment_rotation_halflife = 0.2f;
    float adjustment_position_max_ratio = 0.5f;
    float adjustment_rotation_max_ratio = 0.5f;
    
    // Clamping
    
    bool clamping_enabled = true;
    float clamping_max_distance = 0.15f;
    float clamping_max_angle = 0.5f * PIf;
    
    // IK
    
    static constexpr bool ik_enabled = false;
    float ik_max_length_buffer = 0.015f;
    float ik_foot_height = 0.02f;
    float ik_toe_length = 0.15f;
    float ik_unlock_radius = 0.2f;
    float ik_blending_halflife = 0.1f;
    
    // Learned Motion Matching
    
    static constexpr bool lmm_enabled = false;
    
    // These objects keep Holden's dormant learned path type-correct, but the
    // incompatible LAFAN networks are deliberately not loaded for G1.
    nnet decompressor, stepper, projector;
    nnet_evaluation decompressor_evaluation, stepper_evaluation, projector_evaluation;

    array1d<float> features_proj = db.features(state.frame_index);
    array1d<float> features_curr = db.features(state.frame_index);
    array1d<float> latent_proj(32); latent_proj.zero();
    array1d<float> latent_curr(32); latent_curr.zero();
    
    // Go

    const float dt = 1.0f / 25.0f;
    const float trajectory_sample_time = 1.0f / 3.0f;

#ifdef MM_DISCRETE
    // Optional env overrides so we can sweep halflife without recompiling.
    if (const char* e = getenv("MM_HALFLIFE"))  inertialize_blending_halflife = atof(e);
    if (const char* e = getenv("MM_SIMROT_HL")) simulation_rotation_halflife  = atof(e);
    if (const char* e = getenv("MM_STRAFE"))    g_force_strafe = (atoi(e) != 0);
    const char* logpath = getenv("MM_LOG");
    g_log = fopen(logpath ? logpath : "/home/ubuntu/projects/motion-matching/discrete_log.txt", "w");
    if (!g_log) g_log = stderr;
    fprintf(g_log, "# MM_DISCRETE run: hold-forward + azimuth snaps at f=120,240,360\n");
    fprintf(g_log, "# inertialize_blending_halflife=%.3f sim_rot_halflife=%.3f strafe=%d\n",
        inertialize_blending_halflife, simulation_rotation_halflife, (int)g_force_strafe);
#endif

    int rendered_frames = 0;
    g1_runtime_diagnostic_snapshot runtime_snapshot;
    bool runtime_snapshot_ready = false;
    G1SoleDiagnosticState sole_diagnostic_state;
    const G1LegConfig sole_diagnostic_legs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };

    auto controlled_runtime_error = [&](const char* message)
    {
        fprintf(
            stderr,
            "G1 controlled runtime error scene=%s frame=%d: %s\n",
            active_scene.metadata.id.c_str(),
            state.scene_frame,
            message != NULL ? message : "unknown runtime failure");
        controller_exit_code = 2;
        controller_exit_requested = true;
    };

    motion_match_log deterministic_log;
#ifndef MM_DISCRETE
    const char* deterministic_log_path = getenv("MM_LOG");
#else
    // MM_DISCRETE owns MM_LOG for its legacy text diagnostics.
    const char* deterministic_log_path = NULL;
#endif
    if (!controller_exit_requested &&
        deterministic_log_path != NULL && lmm_enabled) {
        fprintf(stderr,
            "G1 runtime log error: database-frame logging is unavailable "
            "with learned motion matching\n");
        controller_exit_code = 2;
        controller_exit_requested = true;
    }
    if (!controller_exit_requested && !deterministic_log.open(
            deterministic_log_path,
            artifact_error, (int)sizeof(artifact_error))) {
        fprintf(stderr, "G1 runtime log error: %s\n", artifact_error);
        controller_exit_code = 2;
        controller_exit_requested = true;
    }
    const bool logging_enabled = deterministic_log.file != NULL;

    auto update_func = [&]()
    {
        if (rendering_enabled && ::IsKeyPressed(KEY_M))
            show_g1_mesh = !show_g1_mesh;
        if (rendering_enabled && ::IsKeyPressed(KEY_B))
            show_g1_bones = !show_g1_bones;

        if (pending_reset)
        {
            if (!scene_reset_current(
                    state,
                    db,
                    support_rows,
                    active_scene,
                    artifact_error,
                    static_cast<int>(sizeof(artifact_error))))
            {
                controlled_runtime_error(artifact_error);
            }
            else
            {
                ++scene_generation;
                ++scene_reset_count;
                configure_route_cursor(state);
                motion_bank_state_reset(bank_state);
                coverage_safe_stop_latched = false;
            }
            pending_reset = false;
            if (controller_exit_requested) return;
        }
        if (pending_scene_index >= 0 && !controller_exit_requested)
        {
            const int target = pending_scene_index;
            pending_scene_index = -1;
            if (!scene_switch_transaction(
                    active_scene,
                    state,
                    terrain_model,
                    active_scene_index,
                    target,
                    db,
                    support_rows,
                    scene_loader,
                    model_loader,
                    model_unloader,
                    artifact_error,
                    static_cast<int>(sizeof(artifact_error))))
            {
                scene_switch_failed = true;
                fprintf(
                    stderr,
                    "G1 scene switch preserved '%s'; candidate '%s' failed: %s\n",
                    active_scene.metadata.id.c_str(),
                    catalog.ids[static_cast<size_t>(target)].c_str(),
                    artifact_error);
            }
            else
            {
                configure_route_cursor(state);
                motion_bank_state_reset(bank_state);
                coverage_safe_stop_latched = false;
                ++scene_generation;
                ++scene_reset_count;
            }
        }

        state.transitioned = false;
        state.adjustment_xz = 0.0f;
        state.adjustment_y = 0.0f;
        state.clamp_xz = 0.0f;
        state.clamp_y = 0.0f;

#ifdef MM_DISCRETE
        if (test_config.mode == G1_TestLive) {
        // Camera-azimuth scripting. MM_MODE selects the pattern:
        //   0 (default): a few big 90-deg snaps (arrow taps)
        //   1: rapid alternating +/-90 snaps every N frames (arrow mashing)
        //   2: continuous azimuth ramp (arrow held), 2 rad/s like the real cam
        static int mode = -2;
        static int snapN = 12;
        if (mode == -2) { const char* m=getenv("MM_MODE"); mode=m?atoi(m):0;
                          const char* n=getenv("MM_SNAPN"); if(n) snapN=atoi(n); }
        if (mode == 0)
        {
            if (g_frame == 120) state.camera_azimuth += 0.5f * PIf;
            if (g_frame == 240) state.camera_azimuth += 0.5f * PIf;
            if (g_frame == 360) state.camera_azimuth -= 0.5f * PIf;
        }
        else if (mode == 1)
        {
            if (g_frame >= 60 && (g_frame % snapN) == 0)
                state.camera_azimuth += ((g_frame / snapN) % 2 ? -1.0f : 1.0f) * 0.5f * PIf;
        }
        else if (mode == 2)
        {
            if (g_frame >= 60) state.camera_azimuth += 2.0f * (1.0f/60.0f); // arrow held
        }
        else if (mode == 3)
        {
            // alternating 180-deg azimuth snaps -> antipodal state.desired_rotation
            if (g_frame >= 60 && (g_frame % snapN) == 0) state.camera_azimuth += PIf;
        }
        }
#endif

        deterministic_route_sample route_sample;
        route_sample.waypoint = state.route_waypoint;

        // Get gamepad stick states
        vec3 gamepadstick_left = rendering_enabled
            ? gamepad_get_stick(GAMEPAD_STICK_LEFT)
            : vec3(0.0f, 0.0f, 0.9f);
        vec3 gamepadstick_right = rendering_enabled
            ? gamepad_get_stick(GAMEPAD_STICK_RIGHT)
            : vec3();

        // Get if strafe is desired
        bool desired_strafe = rendering_enabled
            ? desired_strafe_update() : false;
#ifdef MM_DISCRETE
        desired_strafe = g_force_strafe;
#endif
        if (test_config.mode != G1_TestLive) {
            desired_strafe = false;
        }
        
        // Get the desired gait (walk / run)
        if (test_config.mode != G1_TestLive) {
            state.desired_gait = 0.0f;
            state.desired_gait_velocity = 0.0f;
        } else {
            desired_gait_update(
                state.desired_gait,
                state.desired_gait_velocity,
                dt);
        }
        
        // Get the desired simulation speeds based on the gait
        float simulation_fwrd_speed = lerpf(simulation_run_fwrd_speed, simulation_walk_fwrd_speed, state.desired_gait);
        float simulation_side_speed = lerpf(simulation_run_side_speed, simulation_walk_side_speed, state.desired_gait);
        float simulation_back_speed = lerpf(simulation_run_back_speed, simulation_walk_back_speed, state.desired_gait);
        
        // Get the desired velocity
        vec3 desired_velocity_curr = desired_velocity_update(
            gamepadstick_left,
            state.camera_azimuth,
            state.simulation_rotation,
            simulation_fwrd_speed,
            simulation_side_speed,
            simulation_back_speed);
        if (test_config.mode == G1_TestRoute) {
            if (state.route_index < 0 ||
                state.route_index >=
                    static_cast<int>(active_scene.metadata.routes.size()) ||
                !deterministic_route_command(
                    route_sample,
                    active_scene.metadata.routes[
                        static_cast<size_t>(state.route_index)],
                    state.route_frames,
                    dt,
                    0.50f,
                    artifact_error,
                    static_cast<int>(sizeof(artifact_error))))
            {
                controlled_runtime_error(
                    state.route_index < 0 ||
                    state.route_index >=
                        static_cast<int>(active_scene.metadata.routes.size())
                        ? "resolved route index is out of range"
                        : artifact_error);
                return;
            }
            desired_velocity_curr = route_sample.command;
            desired_strafe = false;
            state.route_waypoint = route_sample.waypoint;
        }
        const vec3 commanded_velocity = desired_velocity_curr;

        // Heading is selected from the requested command before terrain is
        // allowed to limit travel.  The terrain path never owns this value.
        quat desired_rotation_curr = desired_rotation_update(
            state.desired_rotation,
            gamepadstick_left,
            gamepadstick_right,
            state.camera_azimuth,
            desired_strafe,
            commanded_velocity);
        if (test_heading.active) {
            desired_rotation_curr = test_heading.heading;
        }

        traversability_diagnostics traversal = {};
        desired_velocity_curr = traversability_limit_command(
            state.traversal_speed_scale,
            state.traversal_speed_scale_velocity,
            traversal,
            active_scene.walkability,
            active_scene.terrain,
            state.simulation_position,
            commanded_velocity,
            dt);
        if (!g1_runtime_traversal_is_finite(traversal) ||
            !terrain_float_is_finite(desired_velocity_curr.x) ||
            !terrain_float_is_finite(desired_velocity_curr.y) ||
            !terrain_float_is_finite(desired_velocity_curr.z))
        {
            controlled_runtime_error(
                "command traversal produced non-finite diagnostics");
            return;
        }
        traversability_stop_blocked_planar_dynamics(
            traversal,
            state.simulation_velocity,
            state.simulation_acceleration);
        state.blocked = traversal.blocked;
        state.walkability_class = traversal.walkability_class;
        state.blocked_distance = traversal.distance;
        state.blocked_point = traversal.point;
        
        // Check if we should force a search because input changed quickly
        state.desired_velocity_change_prev = state.desired_velocity_change_curr;
        state.desired_velocity_change_curr =  (desired_velocity_curr - state.desired_velocity) / dt;
        state.desired_velocity = desired_velocity_curr;
        g1_controller_state_seed_first_frame_desired_velocity(state);
        
        state.desired_rotation_change_prev = state.desired_rotation_change_curr;
        state.desired_rotation_change_curr = quat_to_scaled_angle_axis(quat_abs(quat_mul_inv(desired_rotation_curr, state.desired_rotation))) / dt;
        state.desired_rotation =  desired_rotation_curr;

        bool force_search = false;

        if (state.force_search_timer <= 0.0f && (
            (length(state.desired_velocity_change_prev) >= desired_velocity_change_threshold &&
             length(state.desired_velocity_change_curr)  < desired_velocity_change_threshold)
        ||  (length(state.desired_rotation_change_prev) >= desired_rotation_change_threshold &&
             length(state.desired_rotation_change_curr)  < desired_rotation_change_threshold)))
        {
            force_search = true;
            state.force_search_timer = state.search_time;
        }
        else if (state.force_search_timer > 0)
        {
            state.force_search_timer -= dt;
        }
        
        // Predict and publish the complete future trajectory transactionally.
        G1CommandIntent command_intent;
        command_intent.requested_velocity = commanded_velocity;
        command_intent.desired_heading = desired_rotation_curr;

        G1CommandFramePrediction frame_seed = {};
        frame_seed.command = state.command;
        for (int index = 0;
             index < G1CommandTrajectorySampleCount;
             ++index) {
            frame_seed.command.predicted_desired_velocities[index] =
                state.trajectory_desired_velocities(index);
            frame_seed.command.predicted_root_positions[index] =
                state.trajectory_positions(index);
            frame_seed.command.predicted_root_rotations[index] =
                state.trajectory_rotations(index);
            frame_seed.command.predicted_desired_headings[index] =
                state.trajectory_desired_rotations(index);
            frame_seed.predicted_root_velocities[index] =
                state.trajectory_velocities(index);
            frame_seed.predicted_root_accelerations[index] =
                state.trajectory_accelerations(index);
            frame_seed.predicted_root_angular_velocities[index] =
                state.trajectory_angular_velocities(index);
        }

        G1CommandFramePredictionRequest frame_request;
        frame_request.route_mode = test_config.mode == G1_TestRoute;
        frame_request.heading_override = test_heading;
        frame_request.intent = command_intent;
        frame_request.applied_velocity = desired_velocity_curr;

        G1CommandFramePrediction frame_prediction;
        if (!g1_command_frame_prediction_build(
                frame_prediction,
                frame_seed,
                frame_request,
                [&](slice1d<vec3> desired_velocities,
                    bool& route_force_search,
                    char* callback_error,
                    int callback_capacity) {
                    deterministic_route_prediction prediction;
                    if (!deterministic_route_predict_commands(
                            prediction,
                            active_scene.metadata.routes[
                                static_cast<size_t>(state.route_index)],
                            state.route_frames,
                            desired_velocity_curr,
                            dt,
                            0.50f,
                            trajectory_sample_time,
                            state.traversal_speed_scale,
                            false,
                            callback_error,
                            callback_capacity)) {
                        return false;
                    }
                    for (int index = 0;
                         index < G1CommandTrajectorySampleCount;
                         ++index) {
                        desired_velocities(index) = prediction.commands[index];
                    }
                    route_force_search = prediction.force_search;
                    return true;
                },
                [&](slice1d<quat> desired_rotations,
                    const slice1d<vec3> desired_velocities,
                    char*, int) {
                    trajectory_desired_rotations_predict(
                        desired_rotations,
                        desired_velocities,
                        state.desired_rotation,
                        state.camera_azimuth,
                        gamepadstick_left,
                        gamepadstick_right,
                        desired_strafe,
                        trajectory_sample_time);
                    return true;
                },
                [&](slice1d<quat> rotations,
                    slice1d<vec3> angular_velocities,
                    const slice1d<quat> desired_rotations,
                    const slice1d<vec3>,
                    char*, int) {
                    trajectory_rotations_predict(
                        rotations,
                        angular_velocities,
                        state.simulation_rotation,
                        state.simulation_angular_velocity,
                        desired_rotations,
                        simulation_rotation_halflife,
                        trajectory_sample_time);
                    return true;
                },
                [&](slice1d<vec3> desired_velocities,
                    const slice1d<quat> rotations,
                    char*, int) {
                    trajectory_desired_velocities_predict(
                        desired_velocities,
                        rotations,
                        state.desired_velocity,
                        state.camera_azimuth,
                        gamepadstick_left,
                        gamepadstick_right,
                        desired_strafe,
                        simulation_fwrd_speed * state.traversal_speed_scale,
                        simulation_side_speed * state.traversal_speed_scale,
                        simulation_back_speed * state.traversal_speed_scale,
                        trajectory_sample_time);
                    return true;
                },
                [&](slice1d<vec3> positions,
                    slice1d<vec3> velocities,
                    slice1d<vec3> accelerations,
                    const slice1d<vec3> desired_velocities,
                    char*, int) {
                    trajectory_positions_predict(
                        positions,
                        velocities,
                        accelerations,
                        state.simulation_position,
                        state.simulation_velocity,
                        state.simulation_acceleration,
                        desired_velocities,
                        simulation_velocity_halflife,
                        trajectory_sample_time);
                    return true;
                },
                artifact_error,
                static_cast<int>(sizeof(artifact_error)))) {
            controlled_runtime_error(artifact_error);
            return;
        }

        state.command = frame_prediction.command;
        for (int index = 0;
             index < G1CommandTrajectorySampleCount;
             ++index) {
            state.trajectory_desired_velocities(index) =
                frame_prediction.command.predicted_desired_velocities[index];
            state.trajectory_positions(index) =
                frame_prediction.command.predicted_root_positions[index];
            state.trajectory_rotations(index) =
                frame_prediction.command.predicted_root_rotations[index];
            state.trajectory_desired_rotations(index) =
                frame_prediction.command.predicted_desired_headings[index];
            state.trajectory_velocities(index) =
                frame_prediction.predicted_root_velocities[index];
            state.trajectory_accelerations(index) =
                frame_prediction.predicted_root_accelerations[index];
            state.trajectory_angular_velocities(index) =
                frame_prediction.predicted_root_angular_velocities[index];
        }
        force_search = force_search || frame_prediction.force_search;
           
        // Make query vector for search.
        // In theory this only needs to be done when a search is 
        // actually required however for visualization purposes it
        // can be nice to do it every frame
        array1d<float> query(db.nfeatures());
                
        // Compute the features of the query vector

        slice1d<float> query_features = lmm_enabled ? slice1d<float>(features_curr) : db.features(state.frame_index);

        int offset = 0;
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Left Foot Position
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Right Foot Position
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Left Foot Velocity
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Right Foot Velocity
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Hip Velocity
        query_compute_trajectory_position_feature(query, offset, state.bone_positions(0), state.bone_rotations(0), state.trajectory_positions);
        query_compute_trajectory_direction_feature(query, offset, state.bone_rotations(0), state.trajectory_rotations);

        const int query_database_frame = state.frame_index;
        const int query_range = g1_active_range(db, query_database_frame);
        if (active_scene.terrain.version != 2 ||
            !terrain_heightfield_is_queryable(active_scene.terrain) ||
            !terrain_centerline_inputs_are_valid(
                state.bone_positions(0),
                state.trajectory_positions,
                state.trajectory_rotations)) {
            controlled_runtime_error(
                "terrain centerline inputs are invalid");
            return;
        }
        vec3 terrain_centerline_points[
            G1CommandTrajectorySampleCount + 1] = {};
        int terrain_centerline_count = 1;
        terrain_centerline_points[0] = vec3(
            state.bone_positions(0).x, 0.0f,
            state.bone_positions(0).z);
        double terrain_centerline_length = 0.0;
        vec3 terrain_centerline_previous = terrain_centerline_points[0];
        for (int point = 1;
             point < state.trajectory_positions.size;
             ++point)
        {
            const vec3 predicted = state.trajectory_positions(point);
            const double delta_x = static_cast<double>(predicted.x) -
                static_cast<double>(terrain_centerline_previous.x);
            const double delta_z = static_cast<double>(predicted.z) -
                static_cast<double>(terrain_centerline_previous.z);
            const double segment_length = hypot(delta_x, delta_z);
            if (segment_length <= 1e-6) continue;
            terrain_centerline_points[terrain_centerline_count++] = vec3(
                predicted.x, 0.0f, predicted.z);
            terrain_centerline_previous = predicted;
            terrain_centerline_length += segment_length;
        }
        if (terrain_centerline_length < TERRAIN_PROFILE_HORIZON_M)
        {
            terrain_centerline_points[terrain_centerline_count++] =
                terrain_centerline_point_at_arc(
                    state.bone_positions(0),
                    state.trajectory_positions,
                    state.trajectory_rotations,
                    1.125f);
        }
        const slice1d<vec3> terrain_centerline(
            terrain_centerline_count, terrain_centerline_points);
        const vec3 independent_heading =
            terrain_centerline_flattened_heading(
                state.trajectory_desired_rotations(
                    state.trajectory_desired_rotations.size - 1),
                terrain_centerline_flattened_heading(
                    state.desired_rotation, vec3(0.0f, 0.0f, 1.0f)));

        terrain_profile_status terrain_profile_result = terrain_profile_ok;
        terrain_descriptor_v2 terrain_descriptor = {};
        terrain_dense_profile_v2 terrain_dense_profile = {};
        if (!terrain_descriptor_sample_v2(
                terrain_descriptor,
                terrain_profile_result,
                active_scene.terrain,
                terrain_centerline,
                independent_heading) ||
            !terrain_dense_profile_sample_v2(
                terrain_dense_profile,
                terrain_profile_result,
                active_scene.terrain,
                terrain_centerline))
        {
            snprintf(
                artifact_error,
                sizeof(artifact_error),
                "terrain descriptor/profile sampling failed: %s",
                terrain_profile_status_name(terrain_profile_result));
            controlled_runtime_error(artifact_error);
            return;
        }

        motion_bank_classification bank_classification = {};
        motion_bank_classification_status bank_classification_result =
            motion_bank_classification_ok;
        if (!motion_bank_classify_profile(
                bank_classification,
                bank_classification_result,
                terrain_dense_profile.distances_m,
                terrain_dense_profile.heights_m,
                terrain_dense_profile.normals,
                TERRAIN_DENSE_PROFILE_SAMPLE_COUNT))
        {
            snprintf(
                artifact_error,
                sizeof(artifact_error),
                "terrain bank classification failed: %s",
                motion_bank_classification_status_name(
                    bank_classification_result));
            controlled_runtime_error(artifact_error);
            return;
        }
        motion_bank_state_observe(
            bank_state, true, bank_classification);
        if (bank_state.transitioned) force_search = true;

        const motion_family_bank* active_motion_bank =
            motion_bank_for_family(
                motion_manifest,
                motion_bank_family_name(bank_state.current_family));
        if (active_motion_bank == NULL ||
            active_motion_bank->range_indices.empty())
        {
            controlled_runtime_error(
                "terrain classifier did not resolve a nonempty motion bank");
            return;
        }

        uint16_t direction_mask = MOTION_DIRECTION_IDLE;
        uint8_t speed_mask = MOTION_SPEED_LOW;
        const int elevation_mode = bank_state.current_elevation_mode;
        if (!g1_predicted_motion_request_masks(
                direction_mask,
                speed_mask,
                state.trajectory_positions,
                state.trajectory_desired_rotations,
                artifact_error,
                static_cast<int>(sizeof(artifact_error))))
        {
            controlled_runtime_error(artifact_error);
            return;
        }

        for (int terrain_feature = 0;
             terrain_feature < TERRAIN_DESCRIPTOR_VALUE_COUNT;
             ++terrain_feature)
        {
            query(offset++) = terrain_descriptor.values[terrain_feature];
        }

        assert(offset == db.nfeatures());
        if (!motion_match_query_is_finite_39d(query)) {
            controlled_runtime_error(
                "expected exactly 39 finite query values");
            return;
        }

        // Check if we reached the end of the current anim
        bool end_of_anim = database_trajectory_index_clamp(db, state.frame_index, 1) == state.frame_index;
        if (test_config.mode == G1_TestSequential && end_of_anim) {
            snprintf(
                artifact_error,
                sizeof(artifact_error),
                "sequential test overrun at database frame %d before "
                "MM_TEST_FRAMES=%d",
                state.frame_index,
                test_config.frame_limit);
            controlled_runtime_error(artifact_error);
            return;
        }
        const bool matching_enabled = test_config.mode != G1_TestSequential;
        const int incumbent_range = g1_active_range(db, state.frame_index);
        const bool incumbent_compatible =
            incumbent_range >= 0 &&
            database_indexed_range_is_selected(
                active_motion_bank->range_indices.data(),
                static_cast<int>(active_motion_bank->range_indices.size()),
                incumbent_range) &&
            database_indexed_frame_is_publishable(
                db,
                motion_index,
                state.frame_index,
                direction_mask,
                speed_mask,
                elevation_mode);
        force_search = force_search || coverage_safe_stop_latched ||
            !incumbent_compatible;
        state.searched = matching_enabled &&
            (force_search || state.search_timer <= 0.0f || end_of_anim);
        state.incumbent_cost = 0.0f;
        state.selected_cost = 0.0f;
        state.selected_terrain_error = 0.0f;
        state.incumbent_cost = end_of_anim || !incumbent_compatible
            ? FLT_MAX : database_frame_cost(db, state.frame_index, query);
        state.selected_cost = state.incumbent_cost;
        state.selected_terrain_error = database_raw_terrain_error(
            db, state.frame_index, query);
        int selected_database_frame = query_database_frame;
        database_indexed_search_result indexed_search_result;
        bool empty_compatible_set = false;
        bool skip_frame_advance = false;
        
        // Do we need to search?
#ifdef MM_DISCRETE
        int   dbg_best_index = state.frame_index;   // -1 == no search this frame
        bool  dbg_did_search = state.searched;
        bool  dbg_did_transition = false;
        quat  dbg_root_before = state.bone_rotations(0);
        quat  dbg_off_before  = state.bone_offset_rotations(0);
        quat  dbg_trns_dst_rot = state.trns_bone_rotations(0);
#endif
        if (state.searched)
        {
            const int prior_index = state.frame_index;
            const float transition_cost =
                g1_idle_match_transition_cost(
                    traversal.commanded_speed,
                    walkability_xz_length(state.simulation_velocity));
            const database_indexed_search_status search_status =
                database_search_indexed(
                    indexed_search_result,
                    db,
                    motion_index,
                    active_motion_bank->range_indices.data(),
                    static_cast<int>(
                        active_motion_bank->range_indices.size()),
                    direction_mask,
                    speed_mask,
                    elevation_mode,
                    query,
                    prior_index,
                    transition_cost,
                    20,
                    20,
                    G1_MOTION_MATCH_MINIMUM_FUTURE_PUBLISHED_FRAMES);
            if (search_status == DATABASE_INDEXED_SEARCH_INVALID)
            {
                controlled_runtime_error(
                    "indexed motion-bank search rejected runtime inputs");
                return;
            }
            if (search_status == DATABASE_INDEXED_SEARCH_EMPTY)
            {
                empty_compatible_set = true;
                coverage_safe_stop_latched = true;
                G1IkSafeStopHandoff safe_stop_handoff;
                if (!g1_ik_safe_stop_handoff(
                        safe_stop_handoff,
                        coverage_safe_stop_latched,
                        state.desired_velocity,
                        artifact_error,
                        static_cast<int>(sizeof(artifact_error))) ||
                    !safe_stop_handoff.cancel_planar_inertia ||
                    !safe_stop_handoff.force_search)
                {
                    controlled_runtime_error(
                        artifact_error[0] != '\0'
                            ? artifact_error
                            : "coverage safe-stop handoff was not latched");
                    return;
                }
                const vec3 applied_velocity = vec3(
                    safe_stop_handoff.applied_velocity.x,
                    safe_stop_handoff.applied_velocity.y,
                    safe_stop_handoff.applied_velocity.z);
                g1_controller_state_publish_coverage_empty_hold(
                    state, applied_velocity);
                traversal.applied_speed = 0.0f;
                force_search = true;
                state.search_timer = 0.0f;
                skip_frame_advance = true;
#ifdef MM_DISCRETE
                dbg_best_index = -1;
#endif
            }
            else
            {
                coverage_safe_stop_latched = false;
                const int best_index = indexed_search_result.index;
                state.selected_cost = indexed_search_result.cost;
                state.selected_terrain_error = database_raw_terrain_error(
                    db, best_index, query);
                const bool accept_alternative =
                    best_index == prior_index ||
                    g1_motion_match_candidate_beats_continuation(
                        incumbent_compatible,
                        indexed_search_result.cost,
                        state.incumbent_cost);
                selected_database_frame = best_index;
                if (!accept_alternative)
                {
                    selected_database_frame = prior_index;
                    state.selected_cost = state.incumbent_cost;
                    state.selected_terrain_error = database_raw_terrain_error(
                        db, prior_index, query);
                }
                else if (best_index != prior_index)
                {
                    state.transitioned = true;
                    state.trns_bone_positions = db.bone_positions(best_index);
                    state.trns_bone_velocities = db.bone_velocities(best_index);
                    state.trns_bone_rotations = db.bone_rotations(best_index);
                    state.trns_bone_angular_velocities =
                        db.bone_angular_velocities(best_index);

                    inertialize_pose_transition(
                        state.bone_offset_positions,
                        state.bone_offset_velocities,
                        state.bone_offset_rotations,
                        state.bone_offset_angular_velocities,
                        state.transition_src_position,
                        state.transition_src_rotation,
                        state.transition_dst_position,
                        state.transition_dst_rotation,
                        state.bone_positions(0),
                        state.bone_velocities(0),
                        state.bone_rotations(0),
                        state.bone_angular_velocities(0),
                        state.curr_bone_positions,
                        state.curr_bone_velocities,
                        state.curr_bone_rotations,
                        state.curr_bone_angular_velocities,
                        state.trns_bone_positions,
                        state.trns_bone_velocities,
                        state.trns_bone_rotations,
                        state.trns_bone_angular_velocities);

                    state.frame_index = best_index;
#ifdef MM_DISCRETE
                    dbg_did_transition = true;
                    dbg_trns_dst_rot = state.trns_bone_rotations(0);
#endif
                }
#ifdef MM_DISCRETE
                dbg_best_index = selected_database_frame;
#endif
                state.search_timer = state.search_time;
            }
        }
        
        // Tick down search timer
        if (!skip_frame_advance)
        {
            state.search_timer -= dt;
            // Tick frame
            state.frame_index = database_trajectory_index_clamp(
                db, state.frame_index, 1);
            
            // Look-up Next Pose
            state.curr_bone_positions = db.bone_positions(state.frame_index);
            state.curr_bone_velocities = db.bone_velocities(state.frame_index);
            state.curr_bone_rotations = db.bone_rotations(state.frame_index);
            state.curr_bone_angular_velocities = db.bone_angular_velocities(state.frame_index);
            state.curr_bone_contacts = db.contact_states(state.frame_index);

            // Publish the next inertialized pose only after a compatible frame
            // has been accepted.
            inertialize_pose_update(
                state.bone_positions,
                state.bone_velocities,
                state.bone_rotations,
                state.bone_angular_velocities,
                state.bone_offset_positions,
                state.bone_offset_velocities,
                state.bone_offset_rotations,
                state.bone_offset_angular_velocities,
                state.curr_bone_positions,
                state.curr_bone_velocities,
                state.curr_bone_rotations,
                state.curr_bone_angular_velocities,
                state.transition_src_position,
                state.transition_src_rotation,
                state.transition_dst_position,
                state.transition_dst_rotation,
                inertialize_blending_halflife,
                dt);
        }

        motion_match_pose_diagnostic raw_selected_diagnostic;
        motion_match_pose_diagnostic inertialized_diagnostic;
        array1d<vec3> raw_selected_positions(state.curr_bone_positions);
        array1d<quat> raw_selected_rotations(state.curr_bone_rotations);
        raw_selected_positions(0) = state.bone_positions(0);
        raw_selected_rotations(0) = state.bone_rotations(0);
        raw_selected_diagnostic = g1_pose_diagnostic(
            raw_selected_positions, raw_selected_rotations,
            db.bone_parents, active_scene.terrain);
        inertialized_diagnostic = g1_pose_diagnostic(
            state.bone_positions, state.bone_rotations,
            db.bone_parents, active_scene.terrain);
        if (!g1_pose_diagnostic_is_finite(raw_selected_diagnostic) ||
            !g1_pose_diagnostic_is_finite(inertialized_diagnostic) ||
            !terrain_float_is_finite(state.incumbent_cost) ||
            !terrain_float_is_finite(state.selected_cost) ||
            !terrain_float_is_finite(state.selected_terrain_error))
        {
            controlled_runtime_error(
                "motion costs or pose diagnostics are non-finite");
            return;
        }
        
        // Update Simulation

        // An empty indexed-search result is a full accepted-visible-state
        // hold.  Keep simulation heading, support, adjusted pose, and global
        // pose bit-identical until a compatible frame is accepted.
        if (!skip_frame_advance)
        {
        const vec3 simulation_before = state.simulation_position;
        simulation_positions_update(
            state.simulation_position,
            state.simulation_velocity,
            state.simulation_acceleration,
            state.desired_velocity,
            simulation_velocity_halflife,
            dt);
        const walkability_sweep_result integrated_traversal = traversability_preflight_step(
            simulation_before,
            state.simulation_position,
            state.simulation_velocity,
            state.simulation_acceleration,
            active_scene.walkability,
            active_scene.terrain,
            0.20f);
        if (integrated_traversal.blocked) {
            traversability_apply_sweep_result(
                simulation_before,
                state.simulation_position,
                state.simulation_velocity,
                state.simulation_acceleration,
                traversal,
                integrated_traversal);
        }
        walkability_reason current_reason = walkability_clear;
        const int current_walkability_class = walkability_footprint_class(
            active_scene.walkability,
            active_scene.terrain,
            state.simulation_position.x,
            state.simulation_position.z,
            0.20f,
            current_reason);
        state.blocked = traversal.blocked;
        state.blocked_distance = traversal.distance;
        state.blocked_point = traversal.point;
        state.walkability_class = current_walkability_class;
        if (!g1_runtime_traversal_is_finite(traversal) ||
            state.walkability_class < 0 || state.walkability_class > 2 ||
            !terrain_float_is_finite(state.simulation_position.x) ||
            !terrain_float_is_finite(state.simulation_position.y) ||
            !terrain_float_is_finite(state.simulation_position.z))
        {
            controlled_runtime_error(
                "integrated traversal or simulation state is invalid");
            return;
        }
            
        simulation_rotations_update(
            state.simulation_rotation,
            state.simulation_angular_velocity,
            state.desired_rotation,
            simulation_rotation_halflife,
            dt);

        // Observe the active-scene world level from the fully inertialized,
        // support-local pose. Matching and simulation XZ are complete before
        // this downstream world-Y transform is updated.
        forward_kinematics_full(
            state.global_bone_positions,
            state.global_bone_rotations,
            state.bone_positions,
            state.bone_rotations,
            db.bone_parents);
        if (!support_observation_build_walkable(
                state.support_observation_now,
                support_rows,
                state.frame_index,
                active_scene.terrain,
                active_scene.walkability,
                state.global_bone_positions(G1_Simulation),
                state.global_bone_positions(G1_LeftToe),
                state.global_bone_positions(G1_RightToe),
                state.curr_bone_contacts(0),
                state.curr_bone_contacts(1),
                artifact_error,
                static_cast<int>(sizeof(artifact_error))) ||
            !support_frame_update(
                state.support,
                state.support_observation_now,
                state.transitioned,
                dt,
                artifact_error,
                static_cast<int>(sizeof(artifact_error))))
        {
            controlled_runtime_error(artifact_error);
            return;
        }
        if (!support_observation_is_finite(state.support_observation_now) ||
            !g1_runtime_support_state_is_finite(state.support))
        {
            controlled_runtime_error(
                "support observation or state is non-finite");
            return;
        }
        
        // Synchronization 
        
        if (synchronization_enabled && !skip_frame_advance)
        {
            vec3 synchronized_position = lerp(
                state.simulation_position,
                state.bone_positions(0),
                synchronization_data_factor);
                
            quat synchronized_rotation = quat_nlerp_shortest(
                state.simulation_rotation,
                state.bone_rotations(0),
                synchronization_data_factor);
          
            state.simulation_position = synchronized_position;
            state.simulation_rotation = synchronized_rotation;
            
            inertialize_root_adjust(
                state.bone_offset_positions(0),
                state.transition_src_position,
                state.transition_src_rotation,
                state.transition_dst_position,
                state.transition_dst_rotation,
                state.bone_positions(0),
                state.bone_rotations(0),
                synchronized_position,
                synchronized_rotation);
        }
        
        // Adjustment 
        
        if (!skip_frame_advance &&
            !synchronization_enabled && adjustment_enabled)
        {
            const vec3 before_adjustment =
                state.bone_positions(G1_Simulation);
            vec3 adjusted_position;
            quat adjusted_rotation = state.bone_rotations(G1_Simulation);
            
            if (adjustment_by_velocity_enabled)
            {
                adjusted_position =
                    horizontal_adjust_character_position_by_velocity(
                    before_adjustment,
                    state.bone_velocities(G1_Simulation),
                    state.simulation_position,
                    adjustment_position_max_ratio,
                    adjustment_position_halflife,
                    dt);
                
                adjusted_rotation = adjust_character_rotation_by_velocity(
                    state.bone_rotations(G1_Simulation),
                    state.bone_angular_velocities(G1_Simulation),
                    state.simulation_rotation,
                    adjustment_rotation_max_ratio,
                    adjustment_rotation_halflife,
                    dt);
            }
            else
            {
                adjusted_position = horizontal_adjust_character_position(
                    before_adjustment,
                    state.simulation_position,
                    adjustment_position_halflife,
                    dt);
                
                adjusted_rotation = adjust_character_rotation(
                    state.bone_rotations(G1_Simulation),
                    state.simulation_rotation,
                    adjustment_rotation_halflife,
                    dt);
            }

            state.adjustment_xz =
                horizontal_length(adjusted_position - before_adjustment);
            state.adjustment_y =
                adjusted_position.y - before_adjustment.y;
            inertialize_root_adjust(
                state.bone_offset_positions(G1_Simulation),
                state.transition_src_position,
                state.transition_src_rotation,
                state.transition_dst_position,
                state.transition_dst_rotation,
                state.bone_positions(G1_Simulation),
                state.bone_rotations(G1_Simulation),
                adjusted_position,
                adjusted_rotation);
        }
        
        // Clamping
        
        if (!skip_frame_advance &&
            !synchronization_enabled && clamping_enabled)
        {
            const vec3 before_clamp = state.bone_positions(G1_Simulation);
            vec3 adjusted_position = horizontal_clamp_character_position(
                before_clamp,
                state.simulation_position,
                clamping_max_distance);
            quat adjusted_rotation = state.bone_rotations(G1_Simulation);
            
            adjusted_rotation = clamp_character_rotation(
                adjusted_rotation,
                state.simulation_rotation,
                clamping_max_angle);

            state.clamp_xz =
                horizontal_length(adjusted_position - before_clamp);
            state.clamp_y = adjusted_position.y - before_clamp.y;
            inertialize_root_adjust(
                state.bone_offset_positions(G1_Simulation),
                state.transition_src_position,
                state.transition_src_rotation,
                state.transition_dst_position,
                state.transition_dst_rotation,
                state.bone_positions(G1_Simulation),
                state.bone_rotations(G1_Simulation),
                adjusted_position,
                adjusted_rotation);
        }

        if (state.adjustment_y != 0.0f || state.clamp_y != 0.0f)
        {
            controlled_runtime_error("horizontal-root invariant failed");
            return;
        }

        state.adjusted_bone_rotations = state.bone_rotations;
        support_pose_apply(
            state.adjusted_bone_positions,
            state.bone_positions,
            state.support.height);
        forward_kinematics_full(
            state.global_bone_positions,
            state.global_bone_rotations,
            state.adjusted_bone_positions,
            state.adjusted_bone_rotations,
            db.bone_parents);
        }

        const motion_match_pose_diagnostic rendered_diagnostic =
            g1_pose_diagnostic(
                state.adjusted_bone_positions,
                state.adjusted_bone_rotations,
                db.bone_parents,
                active_scene.terrain);
        if (!g1_pose_diagnostic_is_finite(rendered_diagnostic)) {
            controlled_runtime_error(
                "final support-retargeted pose diagnostic is non-finite");
            return;
        }

        // Stage read-only physical sole diagnostics from the same final
        // global ankle poses consumed by the corrected mesh. The persistent
        // slip history is committed only after the complete runtime row is
        // accepted, so a later diagnostic/log failure remains transactional.
        G1SoleDiagnosticState sole_state_candidate = sole_diagnostic_state;
        G1SoleDiagnosticSnapshot sole_snapshot_candidate;
        const bool sole_contacts[2] = {
            state.support_observation_now.contact[0],
            state.support_observation_now.contact[1],
        };
        const G1SoleDiagnosticStatus sole_status =
            g1_sole_diagnostics_observe(
                sole_snapshot_candidate,
                sole_state_candidate,
                active_scene.terrain,
                state.global_bone_positions,
                state.global_bone_rotations,
                sole_diagnostic_legs,
                sole_contacts,
                scene_generation,
                false,
                artifact_error,
                static_cast<int>(sizeof(artifact_error)));
        if (sole_status != G1SoleDiagnosticOk) {
            controlled_runtime_error(artifact_error);
            return;
        }

        g1_runtime_diagnostic_snapshot snapshot_candidate;
        if (!g1_runtime_diagnostics_build(
                snapshot_candidate,
                motion_manifest,
                state,
                traversal,
                route_sample,
                configured_route_target_height,
                scene_generation,
                scene_reset_count,
                scene_switch_failed,
                motion_pack_load_count,
                model_load_count,
                model_unload_count,
                artifact_error,
                static_cast<int>(sizeof(artifact_error))))
        {
            controlled_runtime_error(artifact_error);
            return;
        }
        if (!g1_runtime_diagnostics_attach_sole(
                snapshot_candidate,
                sole_snapshot_candidate,
                artifact_error,
                static_cast<int>(sizeof(artifact_error))))
        {
            controlled_runtime_error(artifact_error);
            return;
        }
        char query_bits_hex[39 * 8 + 1] = {};
        if (!motion_match_query_bits_hex(
                query_bits_hex, sizeof(query_bits_hex), query)) {
            controlled_runtime_error("cannot serialize the finite 39D query");
            return;
        }
        motion_match_log_row log_row;
        log_row.frame = rendered_frames;
        log_row.fixed_dt = dt;
        log_row.scene_id = active_scene.metadata.id.c_str();
        log_row.mode = test_config.name;
        log_row.route = test_config.route;
        log_row.query_bits_hex = query_bits_hex;
        log_row.query_database_frame = query_database_frame;
        log_row.query_range = query_range;
        log_row.selected_database_frame = selected_database_frame;
        log_row.database_frame = state.frame_index;
        log_row.range = g1_active_range(db, state.frame_index);
        log_row.source_range = g1_active_range(db, selected_database_frame);
        log_row.searched = state.searched;
        log_row.transitioned = state.transitioned;
        log_row.incumbent_cost = state.incumbent_cost;
        log_row.selected_cost = state.selected_cost;
        log_row.selected_terrain_error = state.selected_terrain_error;
        log_row.effective_terrain_weight =
                effective_terrain_weight;
        for (int i = 0; i < TERRAIN_DESCRIPTOR_VALUE_COUNT; ++i) {
            log_row.terrain[i] = terrain_descriptor.values[i];
        }
        for (int i = 0; i < 4; ++i) {
            log_row.terrain_points[i] = terrain_descriptor.center_points[i];
        }
        log_row.raw_selected = raw_selected_diagnostic;
        log_row.inertialized = inertialized_diagnostic;
        log_row.rendered = rendered_diagnostic;
        log_row.hips_inertial_offset_y =
            inertialized_diagnostic.hips_y - raw_selected_diagnostic.hips_y;
        log_row.runtime_root_surface_height = heightfield_sample_v2(
            active_scene.terrain,
            state.bone_positions(0).x,
            state.bone_positions(0).z);
        log_row.runtime_left_toe_surface_height = heightfield_sample_v2(
            active_scene.terrain,
            state.global_bone_positions(G1_LeftToe).x,
            state.global_bone_positions(G1_LeftToe).z);
        log_row.runtime_right_toe_surface_height = heightfield_sample_v2(
            active_scene.terrain,
            state.global_bone_positions(G1_RightToe).x,
            state.global_bone_positions(G1_RightToe).z);
        log_row.adjustment_xz = state.adjustment_xz;
        log_row.adjustment_y = state.adjustment_y;
        log_row.clamp_xz = state.clamp_xz;
        log_row.clamp_y = state.clamp_y;
        log_row.matching_enabled = matching_enabled;
        log_row.adjustment_enabled = adjustment_enabled;
        log_row.clamping_enabled = clamping_enabled;
        log_row.support_retargeting_enabled = true;
        log_row.ik_enabled = ik_enabled;
        log_row.source_name = snapshot_candidate.source_name;
        log_row.source_terrain = snapshot_candidate.source_terrain;
        log_row.source_index = snapshot_candidate.source_index;
        log_row.continuation_cost = snapshot_candidate.continuation_cost;
        log_row.source_root_height = snapshot_candidate.source_root_height;
        log_row.source_left_toe_height =
            snapshot_candidate.source_left_toe_height;
        log_row.source_right_toe_height =
            snapshot_candidate.source_right_toe_height;
        log_row.runtime_support_root_height =
            snapshot_candidate.runtime_support_root_height;
        log_row.runtime_support_left_toe_height =
            snapshot_candidate.runtime_support_left_toe_height;
        log_row.runtime_support_right_toe_height =
            snapshot_candidate.runtime_support_right_toe_height;
        log_row.support_root_delta = snapshot_candidate.support_root_delta;
        log_row.support_left_toe_delta =
            snapshot_candidate.support_left_toe_delta;
        log_row.support_right_toe_delta =
            snapshot_candidate.support_right_toe_delta;
        log_row.support_height = snapshot_candidate.support_height;
        log_row.support_velocity = snapshot_candidate.support_velocity;
        log_row.support_source = snapshot_candidate.support_source;
        log_row.airborne_frames = snapshot_candidate.airborne_frames;
        log_row.left_contact = snapshot_candidate.left_contact;
        log_row.right_contact = snapshot_candidate.right_contact;
        log_row.support_retargeted_hips_y =
            snapshot_candidate.support_retargeted_hips_y;
        log_row.ik_adjusted_hips_y = snapshot_candidate.ik_adjusted_hips_y;
        log_row.simulation_x = snapshot_candidate.simulation_x;
        log_row.simulation_z = snapshot_candidate.simulation_z;
        log_row.walkability_class = snapshot_candidate.walkability_class;
        log_row.blocked = snapshot_candidate.blocked;
        log_row.blocked_reason = snapshot_candidate.blocked_reason;
        log_row.blocked_distance = snapshot_candidate.blocked_distance;
        log_row.blocked_point_x = snapshot_candidate.blocked_point_x;
        log_row.blocked_point_z = snapshot_candidate.blocked_point_z;
        log_row.commanded_speed = snapshot_candidate.commanded_speed;
        log_row.applied_speed = snapshot_candidate.applied_speed;
        log_row.route_waypoint = snapshot_candidate.route_waypoint;
        log_row.route_complete = snapshot_candidate.route_complete;
        log_row.route_target_height = snapshot_candidate.route_target_height;
        log_row.scene_generation = snapshot_candidate.scene_generation;
        log_row.scene_frame = snapshot_candidate.scene_frame;
        log_row.scene_reset_count = snapshot_candidate.scene_reset_count;
        log_row.scene_switch_failed = snapshot_candidate.scene_switch_failed;
        log_row.motion_pack_load_count =
            snapshot_candidate.motion_pack_load_count;
        log_row.model_load_count = snapshot_candidate.model_load_count;
        log_row.model_unload_count = snapshot_candidate.model_unload_count;
        log_row.live_model_count = snapshot_candidate.live_model_count;
        for (int foot = 0; foot < 2; ++foot) {
            for (int probe = 0; probe < 4; ++probe) {
                log_row.sole_clearance[foot][probe] =
                    snapshot_candidate.sole_clearance[foot][probe];
            }
            log_row.sole_minimum_clearance[foot] =
                snapshot_candidate.sole_minimum_clearance[foot];
            log_row.stance_slip[foot] =
                snapshot_candidate.stance_slip[foot];
            log_row.stance_slip_reset[foot] =
                snapshot_candidate.stance_slip_reset[foot];
        }
        log_row.sole_global_minimum_clearance =
            snapshot_candidate.sole_global_minimum_clearance;
        log_row.terrain_root_point = terrain_descriptor.root_point;
        for (int sample = 0;
             sample < TERRAIN_DESCRIPTOR_CENTER_SAMPLE_COUNT;
             ++sample)
        {
            log_row.terrain_center_points[sample] =
                terrain_descriptor.center_points[sample];
        }
        for (int sample = 0;
             sample < TERRAIN_DESCRIPTOR_CORRIDOR_SAMPLE_COUNT;
             ++sample)
        {
            log_row.terrain_left_points[sample] =
                terrain_descriptor.left_points[sample];
            log_row.terrain_right_points[sample] =
                terrain_descriptor.right_points[sample];
        }
        log_row.requested_family = motion_bank_family_name(
            bank_classification.family);
        log_row.active_family = motion_bank_family_name(
            bank_state.current_family);
        log_row.source_family = snapshot_candidate.source_family;
        log_row.direction_mask = direction_mask;
        log_row.speed_mask = speed_mask;
        log_row.elevation_mode = elevation_mode;
        log_row.classifier_confidence = bank_classification.confidence;
        log_row.bank_transition = bank_state.transitioned;
        log_row.bank_transition_reason =
            motion_bank_transition_reason_name(bank_state.reason);
        log_row.eligible_frame_count =
            indexed_search_result.eligible_frame_count;
        log_row.evaluated_frame_count =
            indexed_search_result.evaluated_frame_count;
        log_row.considered_bound_count =
            indexed_search_result.considered_bound_count;
        log_row.skipped_bound_count =
            indexed_search_result.skipped_bound_count;
        log_row.empty_compatible_set = empty_compatible_set;
        if (!deterministic_log.write(
                log_row, artifact_error, static_cast<int>(sizeof(artifact_error))))
        {
            controlled_runtime_error(artifact_error);
            return;
        }
        sole_diagnostic_state = sole_state_candidate;
        runtime_snapshot = snapshot_candidate;
        runtime_snapshot_ready = true;
        scene_switch_failed = false;

        if (test_config.mode == G1_TestRoute) {
            if (state.route_frames == INT_MAX) {
                controlled_runtime_error("route frame counter overflow");
                return;
            }
            ++state.route_frames;
        }
        if (test_config.mode == G1_TestSceneCycle &&
            (runtime_snapshot.scene_frame + 1) %
                    test_config.scene_dwell_frames == 0)
        {
            pending_scene_index = (active_scene_index + 1) % scene_count;
        }
        
#ifdef MM_DISCRETE
        {
            // Per-frame instrumentation. All angles in degrees.
            quat root_q   = state.bone_rotations(0);          // final rendered root rotation
            vec3 root_p   = state.bone_positions(0);
            quat off_q    = state.bone_offset_rotations(0);   // inertialize root ROTATION offset
            vec3 off_av   = state.bone_offset_angular_velocities(0);
            static float  prev_root_yaw = dbg_yaw_deg(root_q);
            static quat   prev_root_q   = root_q;
            float root_yaw   = dbg_yaw_deg(root_q);
            float jump_deg   = dbg_angle_between_deg(prev_root_q, root_q); // full 3D jump
            float des_yaw    = dbg_yaw_deg(state.desired_rotation);
            float sim_yaw    = dbg_yaw_deg(state.simulation_rotation);
            float off_ang    = dbg_quat_angle_deg(off_q);           // magnitude of root offset
            float dst_yaw    = dbg_yaw_deg(state.transition_dst_rotation);
            float src_yaw    = dbg_yaw_deg(state.transition_src_rotation);

            fprintf(g_log,
                "f=%d az=%.1f | rootYaw=%.1f jump3D=%.1f | desYaw=%.1f simYaw=%.1f "
                "| offAng=%.2f offW=%.3f offAV=%.2f | best=%d srch=%d trns=%d "
                "| dstYaw=%.1f srcYaw=%.1f | fi=%d\n",
                g_frame, state.camera_azimuth * 180.0f / PIf,
                root_yaw, jump_deg,
                des_yaw, sim_yaw,
                off_ang, off_q.w, length(off_av),
                dbg_best_index, (int)dbg_did_search, (int)dbg_did_transition,
                dst_yaw, src_yaw, state.frame_index);

            if (jump_deg > 30.0f)
            {
                fprintf(g_log,
                    "  *** ROOT JUMP %.1f deg at f=%d: prevYaw=%.1f -> yaw=%.1f "
                    "offAng=%.2f best=%d trns=%d dstYaw=%.1f srcYaw=%.1f\n",
                    jump_deg, g_frame, prev_root_yaw, root_yaw,
                    off_ang, dbg_best_index, (int)dbg_did_transition, dst_yaw, src_yaw);
            }
            fflush(g_log);

            prev_root_yaw = root_yaw;
            prev_root_q   = root_q;
            g_frame++;
            if (g_frame >= 400) {
                controller_exit_requested = true;
                return;
            }
        }
#endif

        // Contact fixup with foot locking and IK

        if constexpr (ik_enabled)
        {
            for (int i = 0; i < state.contact_bones.size; i++)
            {
                // Find all the relevant bone indices
                int toe_bone = state.contact_bones(i);
                int heel_bone = db.bone_parents(toe_bone);
                int knee_bone = db.bone_parents(heel_bone);
                int hip_bone = db.bone_parents(knee_bone);
                int root_bone = db.bone_parents(hip_bone);
                
                // Compute the world space position for the toe
                state.global_bone_computed.zero();
                
                forward_kinematics_partial(
                    state.global_bone_positions,
                    state.global_bone_rotations,
                    state.global_bone_computed,
                    state.bone_positions,
                    state.bone_rotations,
                    db.bone_parents,
                    toe_bone);
                
                // Update the contact state
                contact_update(
                    state.contact_states(i),
                    state.contact_locks(i),
                    state.contact_positions(i),
                    state.contact_velocities(i),
                    state.contact_points(i),
                    state.contact_targets(i),
                    state.contact_offset_positions(i),
                    state.contact_offset_velocities(i),
                    state.global_bone_positions(toe_bone),
                    state.curr_bone_contacts(i),
                    ik_unlock_radius,
                    ik_foot_height,
                    ik_blending_halflife,
                    dt);
                
                // Ensure contact position never goes through floor
                vec3 contact_position_clamp = state.contact_positions(i);
                contact_position_clamp.y = maxf(contact_position_clamp.y, ik_foot_height);
                
                // Re-compute toe, heel, knee, hip, and root bone positions
                for (int bone : {heel_bone, knee_bone, hip_bone, root_bone})
                {
                    forward_kinematics_partial(
                        state.global_bone_positions,
                        state.global_bone_rotations,
                        state.global_bone_computed,
                        state.bone_positions,
                        state.bone_rotations,
                        db.bone_parents,
                        bone);
                }
                
                // Perform simple two-joint IK to place heel
                ik_two_bone(
                    state.adjusted_bone_rotations(hip_bone),
                    state.adjusted_bone_rotations(knee_bone),
                    state.global_bone_positions(hip_bone),
                    state.global_bone_positions(knee_bone),
                    state.global_bone_positions(heel_bone),
                    contact_position_clamp + (state.global_bone_positions(heel_bone) - state.global_bone_positions(toe_bone)),
                    quat_mul_vec3(state.global_bone_rotations(knee_bone), vec3(0.0f, 1.0f, 0.0f)),
                    state.global_bone_rotations(hip_bone),
                    state.global_bone_rotations(knee_bone),
                    state.global_bone_rotations(root_bone),
                    ik_max_length_buffer);
                
                // Re-compute toe, heel, and knee positions 
                state.global_bone_computed.zero();
                
                for (int bone : {toe_bone, heel_bone, knee_bone})
                {
                    forward_kinematics_partial(
                        state.global_bone_positions,
                        state.global_bone_rotations,
                        state.global_bone_computed,
                        state.adjusted_bone_positions,
                        state.adjusted_bone_rotations,
                        db.bone_parents,
                        bone);
                }
                
                // Rotate heel so toe is facing toward contact point
                ik_look_at(
                    state.adjusted_bone_rotations(heel_bone),
                    state.global_bone_rotations(knee_bone),
                    state.global_bone_rotations(heel_bone),
                    state.global_bone_positions(heel_bone),
                    state.global_bone_positions(toe_bone),
                    contact_position_clamp);
                
                // Re-compute toe and heel positions
                state.global_bone_computed.zero();
                
                for (int bone : {toe_bone, heel_bone})
                {
                    forward_kinematics_partial(
                        state.global_bone_positions,
                        state.global_bone_rotations,
                        state.global_bone_computed,
                        state.adjusted_bone_positions,
                        state.adjusted_bone_rotations,
                        db.bone_parents,
                        bone);
                }
                
                // Rotate toe bone so that the end of the toe 
                // does not intersect with the ground
                vec3 toe_end_curr = quat_mul_vec3(
                    state.global_bone_rotations(toe_bone), vec3(ik_toe_length, 0.0f, 0.0f)) +
                    state.global_bone_positions(toe_bone);
                    
                vec3 toe_end_targ = toe_end_curr;
                toe_end_targ.y = maxf(toe_end_targ.y, ik_foot_height);
                
                ik_look_at(
                    state.adjusted_bone_rotations(toe_bone),
                    state.global_bone_rotations(heel_bone),
                    state.global_bone_rotations(toe_bone),
                    state.global_bone_positions(toe_bone),
                    toe_end_curr,
                    toe_end_targ);
            }
        }
        
        if (rendering_enabled)
        {
        // Update camera

        orbit_camera_update(
            camera, 
            state.camera_azimuth,
            state.camera_altitude,
            state.camera_distance,
            state.adjusted_bone_positions(0) + vec3(0, 1, 0),
            // state.simulation_position + vec3(0, 1, 0),
            gamepadstick_right,
            desired_strafe,
            dt);

        if (!::g1_mesh_renderer_update(
                g1_mesh_renderer,
                state.global_bone_positions,
                state.global_bone_rotations,
                artifact_error,
                static_cast<int>(sizeof(artifact_error))))
        {
            controlled_runtime_error(artifact_error);
            return;
        }

        // Render
        
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        BeginMode3D(camera);

        DrawModel(
            terrain_model,
            Vector3{ 0.0f, 0.0f, 0.0f },
            1.0f,
            Color{ 205, 199, 184, 255 });
        DrawModelWires(
            terrain_model,
            Vector3{ 0.0f, 0.0f, 0.0f },
            1.0f,
            DARKGRAY);

        if (show_g1_mesh)
        {
            ::g1_mesh_renderer_draw(g1_mesh_renderer);
        }

        // Draw Simulation Object
        
        DrawCylinderWires(to_Vector3(state.simulation_position), 0.6f, 0.6f, 0.001f, 17, ORANGE);
        DrawSphereWires(to_Vector3(state.simulation_position), 0.05f, 4, 10, ORANGE);
        DrawLine3D(to_Vector3(state.simulation_position), to_Vector3(
            state.simulation_position + 0.6f * quat_mul_vec3(state.simulation_rotation, vec3(0.0f, 0.0f, 1.0f))), ORANGE);
        
        // Draw Clamping Radius/Angles
        
        if (clamping_enabled)
        {
            DrawCylinderWires(
                to_Vector3(state.simulation_position),
                clamping_max_distance, 
                clamping_max_distance, 
                0.001f, 17, SKYBLUE);
            
            quat rotation_clamp_0 = quat_mul(quat_from_angle_axis(+clamping_max_angle, vec3(0.0f, 1.0f, 0.0f)), state.simulation_rotation);
            quat rotation_clamp_1 = quat_mul(quat_from_angle_axis(-clamping_max_angle, vec3(0.0f, 1.0f, 0.0f)), state.simulation_rotation);
            
            vec3 rotation_clamp_0_dir = state.simulation_position + 0.6f * quat_mul_vec3(rotation_clamp_0, vec3(0.0f, 0.0f, 1.0f));
            vec3 rotation_clamp_1_dir = state.simulation_position + 0.6f * quat_mul_vec3(rotation_clamp_1, vec3(0.0f, 0.0f, 1.0f));

            DrawLine3D(to_Vector3(state.simulation_position), to_Vector3(rotation_clamp_0_dir), SKYBLUE);
            DrawLine3D(to_Vector3(state.simulation_position), to_Vector3(rotation_clamp_1_dir), SKYBLUE);
        }
        
        // Draw IK foot lock positions
        
        if constexpr (ik_enabled)
        {
            for (int i = 0; i <  state.contact_positions.size; i++)
            {
                if (state.contact_locks(i))
                {
                    DrawSphereWires(to_Vector3(state.contact_positions(i)), 0.05f, 4, 10, PINK);
                }
            }
        }
        
        draw_trajectory(
            state.trajectory_positions,
            state.trajectory_rotations,
            ORANGE);
        
        // G1: no skinned mesh — draw the skeleton directly from bone transforms.
        // Sphere at each joint, capsule (cylinder) from each bone to its parent.
        if (show_g1_bones)
        {
            for (int bi = 1; bi < db.nbones(); bi++)
            {
                vec3 bp = state.global_bone_positions(bi);
                DrawSphereWires(to_Vector3(bp), 0.028f, 4, 8, DARKBLUE);
                int par = db.bone_parents(bi);
                if (par > 0)
                {
                    DrawCylinderEx(
                        to_Vector3(state.global_bone_positions(par)),
                        to_Vector3(bp),
                        0.018f,
                        0.018f,
                        6,
                        SKYBLUE);
                }
            }
        }
        
        // Draw matched features
        
        array1d<float> current_features = lmm_enabled ? slice1d<float>(features_curr) : db.features(state.frame_index);
        denormalize_features(current_features, db.features_offset, db.features_scale);        
        draw_features(current_features, state.bone_positions(0), state.bone_rotations(0), MAROON);
        
// (diagnostic MM_LOGROOT block removed)
        // Draw Simuation Bone

        DrawSphereWires(to_Vector3(state.bone_positions(0)), 0.05f, 4, 10, MAROON);
        DrawLine3D(to_Vector3(state.bone_positions(0)), to_Vector3(
            state.bone_positions(0) + 0.6f * quat_mul_vec3(state.bone_rotations(0), vec3(0.0f, 0.0f, 1.0f))), MAROON);
        
        draw_axis(vec3(), quat());

        for (int terrain_sample = 0; terrain_sample < 4; ++terrain_sample)
        {
            vec3 point = terrain_descriptor.center_points[terrain_sample];
            point.y += 0.10f;
            DrawSphereWires(to_Vector3(point), 0.04f, 4, 8, PURPLE);
        }
        if (runtime_snapshot_ready && runtime_snapshot.blocked)
        {
            const vec3 blocked_marker(
                runtime_snapshot.blocked_point_x,
                0.10f,
                runtime_snapshot.blocked_point_z);
            DrawSphereWires(to_Vector3(blocked_marker), 0.08f, 6, 12, RED);
        }

        EndMode3D();

        // UI
        
        //---------

        if (test_config.mode != G1_TestLive) {
            GuiDisable();
        }

        GuiGroupBox(Rectangle{ 330, 20, 610, 170 }, "terrain scene / runtime");
        GuiLabel(
            Rectangle{ 350, 30, 310, 20 },
            TextFormat(
                "%s (%d/%d)",
                active_scene.metadata.id.c_str(),
                active_scene_index + 1,
                static_cast<int>(catalog.ids.size())));
        if (GuiButton(Rectangle{ 670, 30, 80, 20 }, "previous"))
        {
            pending_scene_index =
                (active_scene_index + scene_count - 1) % scene_count;
        }
        if (GuiButton(Rectangle{ 760, 30, 80, 20 }, "next"))
        {
            pending_scene_index = (active_scene_index + 1) % scene_count;
        }
        if (GuiButton(Rectangle{ 850, 30, 70, 20 }, "reset"))
        {
            pending_reset = true;
        }
        if (runtime_snapshot_ready)
        {
            GuiLabel(
                Rectangle{ 350, 55, 570, 20 },
                TextFormat(
                    "weight requested %.3f effective %.3f | CSV %s",
                    requested_terrain_weight,
                    effective_terrain_weight,
                    logging_enabled ? "enabled" : "disabled"));
            GuiLabel(
                Rectangle{ 350, 75, 570, 20 },
                TextFormat(
                    "source %d %s terrain=%s",
                    runtime_snapshot.source_index,
                    runtime_snapshot.source_name,
                    runtime_snapshot.source_terrain));
            GuiLabel(
                Rectangle{ 350, 95, 570, 20 },
                TextFormat(
                    "support h=%.3f v=%.3f source=%s contacts=%d/%d",
                    runtime_snapshot.support_height,
                    runtime_snapshot.support_velocity,
                    runtime_snapshot.support_source,
                    static_cast<int>(runtime_snapshot.left_contact),
                    static_cast<int>(runtime_snapshot.right_contact)));
            GuiLabel(
                Rectangle{ 350, 115, 570, 20 },
                TextFormat(
                    "walkability class=%d blocked=%d reason=%s distance=%.3g",
                    runtime_snapshot.walkability_class,
                    static_cast<int>(runtime_snapshot.blocked),
                    runtime_snapshot.blocked_reason,
                    runtime_snapshot.blocked_distance));
            GuiLabel(
                Rectangle{ 350, 135, 570, 20 },
                TextFormat(
                    "generation=%d frame=%d resets=%d switch_failed=%d",
                    runtime_snapshot.scene_generation,
                    runtime_snapshot.scene_frame,
                    runtime_snapshot.scene_reset_count,
                    static_cast<int>(runtime_snapshot.scene_switch_failed)));
            GuiLabel(
                Rectangle{ 350, 155, 570, 20 },
                TextFormat(
                    "route waypoint=%d complete=%d target=%.3f models=%d",
                    runtime_snapshot.route_waypoint,
                    static_cast<int>(runtime_snapshot.route_complete),
                    runtime_snapshot.route_target_height,
                    runtime_snapshot.live_model_count));
        }
        
        float ui_sim_hei = 20;
        
        GuiGroupBox(Rectangle{ 970, ui_sim_hei, 290, 250 }, "simulation object");

        GuiSliderBar(
            Rectangle{ 1100, ui_sim_hei + 10, 120, 20 },
            "velocity halflife", 
            TextFormat("%5.3f", simulation_velocity_halflife), 
            &simulation_velocity_halflife, 0.0f, 0.5f);
            
        GuiSliderBar(
            Rectangle{ 1100, ui_sim_hei + 40, 120, 20 },
            "rotation halflife", 
            TextFormat("%5.3f", simulation_rotation_halflife), 
            &simulation_rotation_halflife, 0.0f, 0.5f);
            
        GuiSliderBar(
            Rectangle{ 1100, ui_sim_hei + 70, 120, 20 },
            "run forward speed", 
            TextFormat("%5.3f", simulation_run_fwrd_speed), 
            &simulation_run_fwrd_speed, 0.0f, 10.0f);
        
        GuiSliderBar(
            Rectangle{ 1100, ui_sim_hei + 100, 120, 20 },
            "run sideways speed", 
            TextFormat("%5.3f", simulation_run_side_speed), 
            &simulation_run_side_speed, 0.0f, 10.0f);
        
        GuiSliderBar(
            Rectangle{ 1100, ui_sim_hei + 130, 120, 20 },
            "run backwards speed", 
            TextFormat("%5.3f", simulation_run_back_speed), 
            &simulation_run_back_speed, 0.0f, 10.0f);
        
        GuiSliderBar(
            Rectangle{ 1100, ui_sim_hei + 160, 120, 20 },
            "walk forward speed", 
            TextFormat("%5.3f", simulation_walk_fwrd_speed), 
            &simulation_walk_fwrd_speed, 0.0f, 5.0f);
        
        GuiSliderBar(
            Rectangle{ 1100, ui_sim_hei + 190, 120, 20 },
            "walk sideways speed", 
            TextFormat("%5.3f", simulation_walk_side_speed), 
            &simulation_walk_side_speed, 0.0f, 5.0f);
        
        GuiSliderBar(
            Rectangle{ 1100, ui_sim_hei + 220, 120, 20 },
            "walk backwards speed", 
            TextFormat("%5.3f", simulation_walk_back_speed), 
            &simulation_walk_back_speed, 0.0f, 5.0f);
        
        //---------
        
        float ui_inert_hei = 280;
        
        GuiGroupBox(Rectangle{ 970, ui_inert_hei, 290, 40 }, "inertiaization blending");
        
        GuiSliderBar(
            Rectangle{ 1100, ui_inert_hei + 10, 120, 20 },
            "halflife", 
            TextFormat("%5.3f", inertialize_blending_halflife), 
            &inertialize_blending_halflife, 0.0f, 0.3f);
        
        //---------
        
        float ui_lmm_hei = 330;
        
        GuiGroupBox(Rectangle{ 970, ui_lmm_hei, 290, 40 }, "learned motion matching");

        GuiLabel(
            Rectangle{ 990, ui_lmm_hei + 10, 250, 20 },
            "disabled: G1 network integration later");
        
        //---------
        
        float ui_ctrl_hei = 380;
        
        GuiGroupBox(Rectangle{ 970, ui_ctrl_hei, 290, 160 }, "controls");

        GuiLabel(Rectangle{ 990, ui_ctrl_hei +  10, 250, 20 }, "WASD / left stick - move");
        GuiLabel(Rectangle{ 990, ui_ctrl_hei +  35, 250, 20 }, "Arrows / right stick - camera");
        GuiLabel(Rectangle{ 990, ui_ctrl_hei +  60, 250, 20 }, "Left trigger - strafe");
        GuiLabel(Rectangle{ 990, ui_ctrl_hei +  85, 250, 20 }, "Shoulders - zoom");
        GuiLabel(Rectangle{ 990, ui_ctrl_hei + 110, 250, 20 }, "A button - walk");
        GuiLabel(
            Rectangle{ 990, ui_ctrl_hei + 135, 250, 20 },
            TextFormat(
                "M mesh %s | B bones %s",
                show_g1_mesh ? "ON" : "OFF",
                show_g1_bones ? "ON" : "OFF"));
        

        
        //---------
        
        GuiGroupBox(Rectangle{ 20, 20, 290, 280 }, "feature weights / terrain diagnostics");
        
        GuiSliderBar(
            Rectangle{ 150, 30, 120, 20 },
            "foot position", 
            TextFormat("%5.3f", feature_weight_foot_position), 
            &feature_weight_foot_position, 0.001f, 3.0f);
            
        GuiSliderBar(
            Rectangle{ 150, 60, 120, 20 },
            "foot velocity", 
            TextFormat("%5.3f", feature_weight_foot_velocity), 
            &feature_weight_foot_velocity, 0.001f, 3.0f);
        
        GuiSliderBar(
            Rectangle{ 150, 90, 120, 20 },
            "hip velocity", 
            TextFormat("%5.3f", feature_weight_hip_velocity), 
            &feature_weight_hip_velocity, 0.001f, 3.0f);
        
        GuiSliderBar(
            Rectangle{ 150, 120, 120, 20 },
            "trajectory positions", 
            TextFormat("%5.3f", feature_weight_trajectory_positions), 
            &feature_weight_trajectory_positions, 0.001f, 3.0f);
        
        GuiSliderBar(
            Rectangle{ 150, 150, 120, 20 },
            "trajectory directions", 
            TextFormat("%5.3f", feature_weight_trajectory_directions), 
            &feature_weight_trajectory_directions, 0.001f, 3.0f);

        GuiSliderBar(
            Rectangle{ 150, 180, 120, 20 },
            "terrain",
            TextFormat("%5.3f", requested_terrain_weight),
            &requested_terrain_weight, 0.0f, 10.0f);

        GuiLabel(
            Rectangle{ 40, 205, 250, 20 },
            requested_terrain_weight == effective_terrain_weight
                ? TextFormat(
                    "effective terrain %.3f", effective_terrain_weight)
                : TextFormat(
                    "requested %.3f (unapplied %.3f)",
                    requested_terrain_weight,
                    effective_terrain_weight));

        if (GuiButton(Rectangle{ 150, 230, 120, 20 }, "apply / rebuild"))
        {
            database_build_matching_features(
                db,
                feature_weight_foot_position,
                feature_weight_foot_velocity,
                feature_weight_hip_velocity,
                feature_weight_trajectory_positions,
                feature_weight_trajectory_directions,
                G1_LeftAnkle,
                G1_RightAnkle,
                G1_Hips,
                requested_terrain_weight);
            if (!g1_matching_features_validate(
                    db,
                    artifact_error,
                    static_cast<int>(sizeof(artifact_error))))
            {
                controlled_runtime_error(artifact_error);
            }
            else
            {
                effective_terrain_weight = requested_terrain_weight;
            }
        }

        GuiLabel(
            Rectangle{ 40, 255, 250, 20 },
            TextFormat(
                "query frame %d  range %d",
                query_database_frame,
                query_range));
        GuiLabel(
            Rectangle{ 40, 275, 250, 20 },
            TextFormat(
                "terrain %.2f %.2f %.2f %.2f",
                terrain_descriptor.values[0],
                terrain_descriptor.values[1],
                terrain_descriptor.values[2],
                terrain_descriptor.values[3]));
        
        //---------
        
        float ui_sync_hei = 310;
        
        GuiGroupBox(Rectangle{ 20, ui_sync_hei, 290, 70 }, "synchronization");

        GuiCheckBox(
            Rectangle{ 50, ui_sync_hei + 10, 20, 20 },
            "enabled",
            &synchronization_enabled);

        GuiSliderBar(
            Rectangle{ 150, ui_sync_hei + 40, 120, 20 },
            "data-driven amount", 
            TextFormat("%5.3f", synchronization_data_factor), 
            &synchronization_data_factor, 0.0f, 1.0f);

        //---------
        
        float ui_adj_hei = 390;
        
        GuiGroupBox(Rectangle{ 20, ui_adj_hei, 290, 130 }, "adjustment");
        
        GuiCheckBox(
            Rectangle{ 50, ui_adj_hei + 10, 20, 20 },
            "enabled",
            &adjustment_enabled);    
        
        GuiCheckBox(
            Rectangle{ 50, ui_adj_hei + 40, 20, 20 },
            "clamp to max velocity",
            &adjustment_by_velocity_enabled);    
        
        GuiSliderBar(
            Rectangle{ 150, ui_adj_hei + 70, 120, 20 },
            "position halflife", 
            TextFormat("%5.3f", adjustment_position_halflife), 
            &adjustment_position_halflife, 0.0f, 0.5f);
        
        GuiSliderBar(
            Rectangle{ 150, ui_adj_hei + 100, 120, 20 },
            "rotation halflife", 
            TextFormat("%5.3f", adjustment_rotation_halflife), 
            &adjustment_rotation_halflife, 0.0f, 0.5f);
        
        //---------
        
        float ui_clamp_hei = 530;
        
        GuiGroupBox(Rectangle{ 20, ui_clamp_hei, 290, 100 }, "clamping");
        
        GuiCheckBox(
            Rectangle{ 50, ui_clamp_hei + 10, 20, 20 },
            "enabled",
            &clamping_enabled);      
        
        GuiSliderBar(
            Rectangle{ 150, ui_clamp_hei + 40, 120, 20 },
            "distance", 
            TextFormat("%5.3f", clamping_max_distance), 
            &clamping_max_distance, 0.0f, 0.5f);
        
        GuiSliderBar(
            Rectangle{ 150, ui_clamp_hei + 70, 120, 20 },
            "angle", 
            TextFormat("%5.3f", clamping_max_angle), 
            &clamping_max_angle, 0.0f, PIf);
        
        //---------
        
        float ui_ik_hei = 640;

        GuiGroupBox(Rectangle{ 20, ui_ik_hei, 290, 40 }, "inverse kinematics");
        GuiLabel(
            Rectangle{ 40, ui_ik_hei + 10, 250, 20 },
            "world-Y support enabled; IK disabled");
        
        //---------

        if (test_config.mode != G1_TestLive) {
            GuiEnable();
        }

        EndDrawing();
        }

        ++state.scene_frame;
        ++rendered_frames;
        if (test_config.frame_limit > 0 &&
            rendered_frames >= test_config.frame_limit)
        {
            controller_exit_requested = true;
        }

    };

    bool cleanup_complete = false;
    auto normal_cleanup = [&]()
    {
        if (cleanup_complete) return;
        cleanup_complete = true;

#ifdef MM_DISCRETE
        if (g_log != NULL && g_log != stderr)
        {
            bool discrete_log_ok = fflush(g_log) == 0;
            if (fclose(g_log) != 0) discrete_log_ok = false;
            g_log = NULL;
            if (!discrete_log_ok)
            {
                fprintf(
                    stderr,
                    "G1 discrete log error during normal cleanup\n");
                if (controller_exit_code == 0) controller_exit_code = 2;
            }
        }
#endif

        const bool log_evidence_ok = deterministic_log.close(
            artifact_error, (int)sizeof(artifact_error));
        const bool log_closed = true;
        if (!log_evidence_ok) {
            fprintf(stderr, "G1 runtime log error: %s\n", artifact_error);
            if (controller_exit_code == 0) controller_exit_code = 2;
        }
        ::g1_mesh_renderer_unload(g1_mesh_renderer);
        model_unloader(terrain_model);

        if (rendering_enabled) CloseWindow();
        const bool window_closed = true;

        cleanup_report cleanup;
        cleanup.exit_code = controller_exit_code;
        cleanup.motion_pack_load_count = motion_pack_load_count;
        cleanup.model_load_count = model_load_count;
        cleanup.model_unload_count = model_unload_count;
        cleanup.log_closed = log_closed;
        cleanup.window_closed = window_closed;
        if (!cleanup_report_write(
                getenv("MM_CLEANUP_LOG"),
                cleanup,
                artifact_error,
                static_cast<int>(sizeof(artifact_error))))
        {
            fprintf(stderr, "G1 cleanup report error: %s\n", artifact_error);
            if (controller_exit_code == 0) controller_exit_code = 2;
        }
    };

#if defined(PLATFORM_WEB)
    std::function<void()> u{[&]()
    {
        const bool window_close_requested = WindowShouldClose();
        if (!controller_exit_requested && !window_close_requested)
        {
            update_func();
        }
        if (controller_exit_requested || window_close_requested)
        {
            normal_cleanup();
            emscripten_cancel_main_loop();
        }
    }};
    emscripten_set_main_loop_arg(update_callback, &u, 0, 1);
    // simulate_infinite_loop normally does not return; clean up defensively if
    // a platform implementation does.
    normal_cleanup();
#else
    while (!controller_exit_requested &&
           (!rendering_enabled || !WindowShouldClose()))
    {
        update_func();
    }
    if (test_config.mode != G1_TestLive &&
        !controller_exit_requested &&
        rendered_frames < test_config.frame_limit)
    {
        fprintf(stderr,
            "G1 deterministic test window closed after %d of %d frames\n",
            rendered_frames, test_config.frame_limit);
        controller_exit_code = 2;
    }
    normal_cleanup();
#endif

    return controller_exit_code;
}
