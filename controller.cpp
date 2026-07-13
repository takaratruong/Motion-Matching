#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#include "raylib.h"
#include "raymath.h"
#include "common.h"
#include "vec.h"
#include "quat.h"
#include "spring.h"
#include "array.h"
#include "character.h"
#include "database.h"
#include "g1_skeleton.h"
#include "terrain_runtime.h"
#include "motion_match_log.h"
#include "nnet.h"
#include "lmm.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <initializer_list>
#include <functional>

//--------------------------------------

static inline Vector3 to_Vector3(vec3 v)
{
    return (Vector3){ v.x, v.y, v.z };
}

static bool g1_artifact_path(
    char* output,
    const size_t capacity,
    const char* directory,
    const char* filename,
    char* error,
    const int error_capacity)
{
    if (directory == NULL || directory[0] == '\0')
    {
        return g1_error(
            error, error_capacity, "G1_TERRAIN_DIR must not be empty");
    }

    const int length = snprintf(output, capacity, "%s/%s", directory, filename);
    if (length < 0 || static_cast<size_t>(length) >= capacity)
    {
        return g1_error(
            error,
            error_capacity,
            "G1 terrain path is too long for %s",
            filename);
    }
    return true;
}

static bool g1_probe_required_file(
    const char* path, char* error, const int error_capacity)
{
    FILE* file = fopen(path, "rb");
    if (file == NULL)
    {
        return g1_error(
            error,
            error_capacity,
            "%s: cannot open required G1 terrain artifact (%s)",
            path,
            strerror(errno));
    }
    if (fseek(file, 0, SEEK_END) != 0)
    {
        fclose(file);
        return g1_error(
            error, error_capacity, "%s: cannot size required artifact", path);
    }
    const long size = ftell(file);
    const bool close_failed = fclose(file) != 0;
    if (size <= 0 || close_failed)
    {
        return g1_error(
            error,
            error_capacity,
            "%s: required artifact is empty or unreadable",
            path);
    }
    return true;
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

enum g1_test_mode { G1_TestLive, G1_TestSequential, G1_TestFlat, G1_TestTerrain };

struct g1_test_config
{
    g1_test_mode mode = G1_TestLive;
    const char* name = "live";
    const char* route = "manual";
    int frame_limit = 0;
};

static bool g1_parse_test_config(
    g1_test_config& out, char* error, const int error_capacity)
{
    const char* mode = getenv("MM_TEST_MODE");
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
        } else {
            return g1_error(error, error_capacity,
                "MM_TEST_MODE must be sequential, flat, or terrain, got '%s'",
                mode);
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
    out.hips_clearance = positions(G1_Hips).y - heightfield_sample(
        terrain, positions(G1_Hips).x, positions(G1_Hips).z);
    out.left_toe_clearance = positions(G1_LeftToe).y - heightfield_sample(
        terrain, positions(G1_LeftToe).x, positions(G1_LeftToe).z);
    out.right_toe_clearance = positions(G1_RightToe).y - heightfield_sample(
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
            p.y - heightfield_sample(terrain, p.x, p.z));
    }
    return out;
}

static float g1_xz_length(const vec3 value)
{
    return sqrtf(value.x * value.x + value.z * value.z);
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
    const int expected_features = 31;
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
    Mesh mesh = { 0 };
    
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

    cam.target = (Vector3){ target.x, target.y, target.z };
    cam.position = (Vector3){ eye.x, eye.y, eye.z };
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
    
    // If strafe is not active the desired direction comes from the left 
    // stick as long as that stick is being used
    else if (length(gamepadstick_left) > 0.01f)
    {
        
        vec3 desired_direction = normalize(desired_velocity);
        return quat_from_angle_axis(atan2f(desired_direction.x, desired_direction.z), vec3(0, 1, 0));
    }
    
    // Otherwise desired direction remains the same
    else
    {
        return desired_rotation_curr;
    }
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

vec3 adjust_character_position(
    const vec3 character_position,
    const vec3 simulation_position,
    const float halflife,
    const float dt)
{
    // Find the difference in positioning
    vec3 difference_position = simulation_position - character_position;
    
    // Damp that difference using the given halflife and dt
    vec3 adjustment_position = damp_adjustment_exact(
        difference_position,
        halflife,
        dt);
    
    // Add the damped difference to move the character toward the sim
    return adjustment_position + character_position;
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

vec3 adjust_character_position_by_velocity(
    const vec3 character_position,
    const vec3 character_velocity,
    const vec3 simulation_position,
    const float max_adjustment_ratio,
    const float halflife,
    const float dt)
{
    // Find and damp the desired adjustment
    vec3 adjustment_position = damp_adjustment_exact(
        simulation_position - character_position,
        halflife,
        dt);
    
    // If the length of the adjustment is greater than the character velocity 
    // multiplied by the ratio then we need to clamp it to that length
    float max_length = max_adjustment_ratio * length(character_velocity) * dt;
    
    if (length(adjustment_position) > max_length)
    {
        adjustment_position = max_length * normalize(adjustment_position);
    }
    
    // Apply the adjustment
    return adjustment_position + character_position;
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

vec3 clamp_character_position(
    const vec3 character_position,
    const vec3 simulation_position,
    const float max_distance)
{
    // If the character deviates too far from the simulation 
    // position we need to clamp it to within the max distance
    if (length(character_position - simulation_position) > max_distance)
    {
        return max_distance * 
            normalize(character_position - simulation_position) + 
            simulation_position;
    }
    else
    {
        return character_position;
    }
}
  
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
    char database_path[1024] = {};
    char feature_path[1024] = {};
    char heightfield_path[1024] = {};
    char mesh_path[1024] = {};
    char manifest_path[1024] = {};
    char artifact_error[512] = {};
    if (!g1_artifact_path(
            database_path,
            sizeof(database_path),
            terrain_directory,
            "database.bin",
            artifact_error,
            sizeof(artifact_error)) ||
        !g1_artifact_path(
            feature_path,
            sizeof(feature_path),
            terrain_directory,
            "terrain_features.bin",
            artifact_error,
            sizeof(artifact_error)) ||
        !g1_artifact_path(
            heightfield_path,
            sizeof(heightfield_path),
            terrain_directory,
            "terrain.bin",
            artifact_error,
            sizeof(artifact_error)) ||
        !g1_artifact_path(
            mesh_path,
            sizeof(mesh_path),
            terrain_directory,
            "terrain.obj",
            artifact_error,
            sizeof(artifact_error)) ||
        !g1_artifact_path(
            manifest_path,
            sizeof(manifest_path),
            terrain_directory,
            "manifest.json",
            artifact_error,
            sizeof(artifact_error)))
    {
        fprintf(stderr, "G1 terrain path error: %s\n", artifact_error);
        return 2;
    }

    const char* required_paths[5] = {
        database_path, feature_path, heightfield_path, mesh_path, manifest_path
    };
    for (int path = 0; path < 5; ++path)
    {
        if (!g1_probe_required_file(
                required_paths[path],
                artifact_error,
                static_cast<int>(sizeof(artifact_error))))
        {
            fprintf(stderr, "G1 terrain artifact error: %s\n", artifact_error);
            return 2;
        }
    }
    if (!g1_manifest_validate(
            manifest_path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 manifest error: %s\n", artifact_error);
        return 2;
    }

    float feature_weight_foot_position = 0.75f;
    float feature_weight_foot_velocity = 1.0f;
    float feature_weight_hip_velocity = 1.0f;
    float feature_weight_trajectory_positions = 1.0f;
    float feature_weight_trajectory_directions = 1.5f;
    float feature_weight_terrain = 0.0f;
    g1_test_config test_config;
    if (!g1_parse_terrain_weight(
            feature_weight_terrain,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        !g1_parse_test_config(
            test_config,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 terrain option error: %s\n", artifact_error);
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
    // Deterministic CSV cleanup cannot run after the Emscripten main loop.
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
        feature_weight_terrain = 0.0f;
    } else if (test_config.mode == G1_TestTerrain &&
               getenv("MM_TERRAIN_WEIGHT") == NULL) {
        feature_weight_terrain = 4.0f;
    }
#ifdef MM_DISCRETE
    if (test_config.mode != G1_TestLive) {
        fprintf(stderr,
            "G1 terrain option error: MM_TEST_MODE is unavailable in "
            "MM_DISCRETE builds\n");
        return 2;
    }
#endif

    terrain_feature_set terrain_rows;
    heightfield runtime_terrain;
    if (!terrain_features_load(
            terrain_rows,
            feature_path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))) ||
        !heightfield_load(
            runtime_terrain,
            heightfield_path,
            artifact_error,
            static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 terrain artifact error: %s\n", artifact_error);
        return 2;
    }

    database db;
    database_load(db, database_path);
    if (!g1_database_validate(
            db, artifact_error, static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 database error: %s\n", artifact_error);
        return 2;
    }
    if (terrain_rows.values.rows != db.nframes())
    {
        fprintf(
            stderr,
            "G1 terrain frame mismatch: database=%d sidecar=%d\n",
            db.nframes(),
            terrain_rows.values.rows);
        return 2;
    }
    db.terrain_features = terrain_rows.values;
    if (!g1_skeleton_validate(
            db, artifact_error, static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 skeleton error: %s\n", artifact_error);
        return 2;
    }
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
        feature_weight_terrain);
    if (!g1_matching_features_validate(
            db, artifact_error, static_cast<int>(sizeof(artifact_error))))
    {
        fprintf(stderr, "G1 feature error: %s\n", artifact_error);
        return 2;
    }
    float applied_feature_weight_terrain = feature_weight_terrain;

    // Open the graphics window only after every artifact and feature gate has
    // passed, so startup errors stay useful on headless systems.
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

    Model terrain_model = LoadModel(mesh_path);
    if (!IsModelReady(terrain_model) || terrain_model.meshCount <= 0)
    {
        fprintf(stderr, "G1 terrain mesh failed to load: %s\n", mesh_path);
        CloseWindow();
        return 2;
    }
    
    // Camera

    Camera3D camera = { 0 };
    camera.position = (Vector3){ 0.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camera_azimuth = 0.0f;
    float camera_altitude = 0.4f;
    float camera_distance = 4.0f;
    
    // Character
    
    // G1: no character.bin skinned mesh — the skeleton is drawn directly from
    // bone transforms in the render loop, so mesh/shader loading is skipped.

    // Pose & Inertializer Data
    
    int frame_index = db.range_starts(0);
    float inertialize_blending_halflife = 0.10f;

    array1d<vec3> curr_bone_positions = db.bone_positions(frame_index);
    array1d<vec3> curr_bone_velocities = db.bone_velocities(frame_index);
    array1d<quat> curr_bone_rotations = db.bone_rotations(frame_index);
    array1d<vec3> curr_bone_angular_velocities = db.bone_angular_velocities(frame_index);
    array1d<bool> curr_bone_contacts = db.contact_states(frame_index);

    array1d<vec3> trns_bone_positions = db.bone_positions(frame_index);
    array1d<vec3> trns_bone_velocities = db.bone_velocities(frame_index);
    array1d<quat> trns_bone_rotations = db.bone_rotations(frame_index);
    array1d<vec3> trns_bone_angular_velocities = db.bone_angular_velocities(frame_index);
    array1d<bool> trns_bone_contacts = db.contact_states(frame_index);

    array1d<vec3> bone_positions = db.bone_positions(frame_index);
    array1d<vec3> bone_velocities = db.bone_velocities(frame_index);
    array1d<quat> bone_rotations = db.bone_rotations(frame_index);
    array1d<vec3> bone_angular_velocities = db.bone_angular_velocities(frame_index);
    
    array1d<vec3> bone_offset_positions(db.nbones());
    array1d<vec3> bone_offset_velocities(db.nbones());
    array1d<quat> bone_offset_rotations(db.nbones());
    array1d<vec3> bone_offset_angular_velocities(db.nbones());
    
    array1d<vec3> global_bone_positions(db.nbones());
    array1d<vec3> global_bone_velocities(db.nbones());
    array1d<quat> global_bone_rotations(db.nbones());
    array1d<vec3> global_bone_angular_velocities(db.nbones());
    array1d<bool> global_bone_computed(db.nbones());
    
    vec3 transition_src_position;
    quat transition_src_rotation;
    vec3 transition_dst_position;
    quat transition_dst_rotation;
    
    inertialize_pose_reset(
        bone_offset_positions,
        bone_offset_velocities,
        bone_offset_rotations,
        bone_offset_angular_velocities,
        transition_src_position,
        transition_src_rotation,
        transition_dst_position,
        transition_dst_rotation,
        bone_positions(0),
        bone_rotations(0));
    
    inertialize_pose_update(
        bone_positions,
        bone_velocities,
        bone_rotations,
        bone_angular_velocities,
        bone_offset_positions,
        bone_offset_velocities,
        bone_offset_rotations,
        bone_offset_angular_velocities,
        db.bone_positions(frame_index),
        db.bone_velocities(frame_index),
        db.bone_rotations(frame_index),
        db.bone_angular_velocities(frame_index),
        transition_src_position,
        transition_src_rotation,
        transition_dst_position,
        transition_dst_rotation,
        inertialize_blending_halflife,
        0.0f);
        
    // Trajectory & Gameplay Data

    float search_time = 0.1f;
#ifdef MM_DISCRETE
    if (const char* e = getenv("MM_SEARCHT")) search_time = atof(e);
#endif
    float search_timer = search_time;
    float force_search_timer = search_time;
    
    vec3 desired_velocity;
    vec3 desired_velocity_change_curr;
    vec3 desired_velocity_change_prev;
    float desired_velocity_change_threshold = 50.0;
    
    quat desired_rotation;
    vec3 desired_rotation_change_curr;
    vec3 desired_rotation_change_prev;
    float desired_rotation_change_threshold = 50.0;
    
    float desired_gait = 0.0f;
    float desired_gait_velocity = 0.0f;
    
    vec3 simulation_position;
    vec3 simulation_velocity;
    vec3 simulation_acceleration;
    quat simulation_rotation;
    vec3 simulation_angular_velocity;
    
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
    
    array1d<vec3> trajectory_desired_velocities(4);
    array1d<quat> trajectory_desired_rotations(4);
    array1d<vec3> trajectory_positions(4);
    array1d<vec3> trajectory_velocities(4);
    array1d<vec3> trajectory_accelerations(4);
    array1d<quat> trajectory_rotations(4);
    array1d<vec3> trajectory_angular_velocities(4);
    
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
    
    const bool ik_enabled = false;
    float ik_max_length_buffer = 0.015f;
    float ik_foot_height = 0.02f;
    float ik_toe_length = 0.15f;
    float ik_unlock_radius = 0.2f;
    float ik_blending_halflife = 0.1f;
    
    // Contact and Foot Locking data
    
    array1d<int> contact_bones(2);
    contact_bones(0) = G1_LeftToe;
    contact_bones(1) = G1_RightToe;
    
    array1d<bool> contact_states(contact_bones.size);
    array1d<bool> contact_locks(contact_bones.size);
    array1d<vec3> contact_positions(contact_bones.size);
    array1d<vec3> contact_velocities(contact_bones.size);
    array1d<vec3> contact_points(contact_bones.size);
    array1d<vec3> contact_targets(contact_bones.size);
    array1d<vec3> contact_offset_positions(contact_bones.size);
    array1d<vec3> contact_offset_velocities(contact_bones.size);
    
    for (int i = 0; i < contact_bones.size; i++)
    {
        vec3 bone_position;
        vec3 bone_velocity;
        quat bone_rotation;
        vec3 bone_angular_velocity;
        
        forward_kinematics_velocity(
            bone_position,
            bone_velocity,
            bone_rotation,
            bone_angular_velocity,
            bone_positions,
            bone_velocities,
            bone_rotations,
            bone_angular_velocities,
            db.bone_parents,
            contact_bones(i));
        
        contact_reset(
            contact_states(i),
            contact_locks(i),
            contact_positions(i),  
            contact_velocities(i),
            contact_points(i),
            contact_targets(i),
            contact_offset_positions(i),
            contact_offset_velocities(i),
            bone_position,
            bone_velocity,
            false);
    }
    
    array1d<vec3> adjusted_bone_positions = bone_positions;
    array1d<quat> adjusted_bone_rotations = bone_rotations;
    
    // Learned Motion Matching
    
    const bool lmm_enabled = false;
    
    // These objects keep Holden's dormant learned path type-correct, but the
    // incompatible LAFAN networks are deliberately not loaded for G1.
    nnet decompressor, stepper, projector;
    nnet_evaluation decompressor_evaluation, stepper_evaluation, projector_evaluation;

    array1d<float> features_proj = db.features(frame_index);
    array1d<float> features_curr = db.features(frame_index);
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
    bool controller_exit_requested = false;
    int controller_exit_code = 0;

    motion_match_log deterministic_log;
#ifndef MM_DISCRETE
    const char* deterministic_log_path = getenv("MM_LOG");
#else
    // MM_DISCRETE owns MM_LOG for its legacy text diagnostics.
    const char* deterministic_log_path = NULL;
#endif
    if (deterministic_log_path != NULL && lmm_enabled) {
        fprintf(stderr,
            "G1 runtime log error: database-frame logging is unavailable "
            "with learned motion matching\n");
        UnloadModel(terrain_model);
        CloseWindow();
        return 2;
    }
    if (!deterministic_log.open(
            deterministic_log_path,
            artifact_error, (int)sizeof(artifact_error))) {
        fprintf(stderr, "G1 runtime log error: %s\n", artifact_error);
        UnloadModel(terrain_model);
        CloseWindow();
        return 2;
    }
    const bool logging_enabled = deterministic_log.file != NULL;

    auto update_func = [&]()
    {

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
            if (g_frame == 120) camera_azimuth += 0.5f * PIf;
            if (g_frame == 240) camera_azimuth += 0.5f * PIf;
            if (g_frame == 360) camera_azimuth -= 0.5f * PIf;
        }
        else if (mode == 1)
        {
            if (g_frame >= 60 && (g_frame % snapN) == 0)
                camera_azimuth += ((g_frame / snapN) % 2 ? -1.0f : 1.0f) * 0.5f * PIf;
        }
        else if (mode == 2)
        {
            if (g_frame >= 60) camera_azimuth += 2.0f * (1.0f/60.0f); // arrow held
        }
        else if (mode == 3)
        {
            // alternating 180-deg azimuth snaps -> antipodal desired_rotation
            if (g_frame >= 60 && (g_frame % snapN) == 0) camera_azimuth += PIf;
        }
        }
#endif

        // Get gamepad stick states
        vec3 gamepadstick_left = gamepad_get_stick(GAMEPAD_STICK_LEFT);
        vec3 gamepadstick_right = gamepad_get_stick(GAMEPAD_STICK_RIGHT);
        if (test_config.mode != G1_TestLive) {
            gamepadstick_left = vec3(0.0f, 0.0f, 0.9f);
            gamepadstick_right = vec3();
        }

        // Get if strafe is desired
        bool desired_strafe = desired_strafe_update();
#ifdef MM_DISCRETE
        desired_strafe = g_force_strafe;
#endif
        if (test_config.mode != G1_TestLive) {
            desired_strafe = false;
        }
        
        // Get the desired gait (walk / run)
        if (test_config.mode != G1_TestLive) {
            desired_gait = 0.0f;
            desired_gait_velocity = 0.0f;
        } else {
            desired_gait_update(
                desired_gait,
                desired_gait_velocity,
                dt);
        }
        
        // Get the desired simulation speeds based on the gait
        float simulation_fwrd_speed = lerpf(simulation_run_fwrd_speed, simulation_walk_fwrd_speed, desired_gait);
        float simulation_side_speed = lerpf(simulation_run_side_speed, simulation_walk_side_speed, desired_gait);
        float simulation_back_speed = lerpf(simulation_run_back_speed, simulation_walk_back_speed, desired_gait);
        
        // Get the desired velocity
        vec3 desired_velocity_curr = desired_velocity_update(
            gamepadstick_left,
            camera_azimuth,
            simulation_rotation,
            simulation_fwrd_speed,
            simulation_side_speed,
            simulation_back_speed);
            
        // Get the desired rotation/direction
        quat desired_rotation_curr = desired_rotation_update(
            desired_rotation,
            gamepadstick_left,
            gamepadstick_right,
            camera_azimuth,
            desired_strafe,
            desired_velocity_curr);
        
        // Check if we should force a search because input changed quickly
        desired_velocity_change_prev = desired_velocity_change_curr;
        desired_velocity_change_curr =  (desired_velocity_curr - desired_velocity) / dt;
        desired_velocity = desired_velocity_curr;
        if (rendered_frames == 0) {
            trajectory_desired_velocities.set(desired_velocity);
        }
        
        desired_rotation_change_prev = desired_rotation_change_curr;
        desired_rotation_change_curr = quat_to_scaled_angle_axis(quat_abs(quat_mul_inv(desired_rotation_curr, desired_rotation))) / dt;
        desired_rotation =  desired_rotation_curr;
        
        bool force_search = false;

        if (force_search_timer <= 0.0f && (
            (length(desired_velocity_change_prev) >= desired_velocity_change_threshold && 
             length(desired_velocity_change_curr)  < desired_velocity_change_threshold)
        ||  (length(desired_rotation_change_prev) >= desired_rotation_change_threshold && 
             length(desired_rotation_change_curr)  < desired_rotation_change_threshold)))
        {
            force_search = true;
            force_search_timer = search_time;
        }
        else if (force_search_timer > 0)
        {
            force_search_timer -= dt;
        }
        
        // Predict Future Trajectory
        
        trajectory_desired_rotations_predict(
          trajectory_desired_rotations,
          trajectory_desired_velocities,
          desired_rotation,
          camera_azimuth,
          gamepadstick_left,
          gamepadstick_right,
          desired_strafe,
          trajectory_sample_time);
        
        trajectory_rotations_predict(
            trajectory_rotations,
            trajectory_angular_velocities,
            simulation_rotation,
            simulation_angular_velocity,
            trajectory_desired_rotations,
            simulation_rotation_halflife,
            trajectory_sample_time);
        
        trajectory_desired_velocities_predict(
          trajectory_desired_velocities,
          trajectory_rotations,
          desired_velocity,
          camera_azimuth,
          gamepadstick_left,
          gamepadstick_right,
          desired_strafe,
          simulation_fwrd_speed,
          simulation_side_speed,
          simulation_back_speed,
          trajectory_sample_time);
        
        trajectory_positions_predict(
            trajectory_positions,
            trajectory_velocities,
            trajectory_accelerations,
            simulation_position,
            simulation_velocity,
            simulation_acceleration,
            trajectory_desired_velocities,
            simulation_velocity_halflife,
            trajectory_sample_time);
           
        // Make query vector for search.
        // In theory this only needs to be done when a search is 
        // actually required however for visualization purposes it
        // can be nice to do it every frame
        array1d<float> query(db.nfeatures());
                
        // Compute the features of the query vector

        slice1d<float> query_features = lmm_enabled ? slice1d<float>(features_curr) : db.features(frame_index);

        int offset = 0;
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Left Foot Position
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Right Foot Position
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Left Foot Velocity
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Right Foot Velocity
        query_copy_denormalized_feature(query, offset, 3, query_features, db.features_offset, db.features_scale); // Hip Velocity
        query_compute_trajectory_position_feature(query, offset, bone_positions(0), bone_rotations(0), trajectory_positions);
        query_compute_trajectory_direction_feature(query, offset, bone_rotations(0), trajectory_rotations);

        const int query_database_frame = frame_index;
        const int query_range = g1_active_range(db, query_database_frame);
        if (!terrain_heightfield_is_queryable(runtime_terrain) ||
            !terrain_centerline_inputs_are_valid(
                bone_positions(0),
                trajectory_positions,
                trajectory_rotations)) {
            fprintf(stderr,
                "G1 runtime query error: terrain centerline inputs are invalid\n");
            controller_exit_code = 2;
            controller_exit_requested = true;
            return;
        }
        terrain_centerline_snapshot terrain_query_snapshot = {};
        terrain_centerline_snapshot_compute(
            terrain_query_snapshot,
            runtime_terrain,
            bone_positions(0),
            trajectory_positions,
            trajectory_rotations);
        for (int terrain_feature = 0; terrain_feature < 4; ++terrain_feature) {
            const vec3 point = terrain_query_snapshot.points[terrain_feature];
            if (!terrain_float_is_finite(
                    terrain_query_snapshot.values[terrain_feature]) ||
                !terrain_float_is_finite(point.x) ||
                !terrain_float_is_finite(point.y) ||
                !terrain_float_is_finite(point.z)) {
                fprintf(stderr,
                    "G1 runtime query error: terrain snapshot %d is invalid\n",
                    terrain_feature);
                controller_exit_code = 2;
                controller_exit_requested = true;
                return;
            }
        }
        for (int terrain_feature = 0; terrain_feature < 4; ++terrain_feature)
        {
            query(offset++) = terrain_query_snapshot.values[terrain_feature];
        }

        assert(offset == db.nfeatures());
        if (!motion_match_query_is_finite_31d(query)) {
            fprintf(stderr,
                "G1 runtime query error: expected 31 finite query values\n");
            controller_exit_code = 2;
            controller_exit_requested = true;
            return;
        }

        // Check if we reached the end of the current anim
        bool end_of_anim = database_trajectory_index_clamp(db, frame_index, 1) == frame_index;
        if (test_config.mode == G1_TestSequential && end_of_anim) {
            fprintf(stderr,
                "G1 sequential test overrun at database frame %d before "
                "MM_TEST_FRAMES=%d\n",
                frame_index, test_config.frame_limit);
            controller_exit_code = 2;
            controller_exit_requested = true;
            return;
        }
        const bool matching_enabled = test_config.mode != G1_TestSequential;
        const bool search_requested = matching_enabled &&
            (force_search || search_timer <= 0.0f || end_of_anim);
        float incumbent_cost = 0.0f;
        float selected_cost = 0.0f;
        float selected_terrain_error = 0.0f;
        if (logging_enabled) {
            incumbent_cost = end_of_anim
                ? FLT_MAX : database_frame_cost(db, frame_index, query);
            selected_cost = incumbent_cost;
            selected_terrain_error = database_raw_terrain_error(
                db, frame_index, query);
        }
        int selected_database_frame = query_database_frame;
        bool transitioned = false;
        
        // Do we need to search?
#ifdef MM_DISCRETE
        int   dbg_best_index = frame_index;   // -1 == no search this frame
        bool  dbg_did_search = search_requested;
        bool  dbg_did_transition = false;
        quat  dbg_root_before = bone_rotations(0);
        quat  dbg_off_before  = bone_offset_rotations(0);
        quat  dbg_trns_dst_rot = trns_bone_rotations(0);
#endif
        if (search_requested)
        {
            if (lmm_enabled)
            {
                // Project query onto nearest feature vector
                
                float best_cost = FLT_MAX;
                bool transition = false;
                
                projector_evaluate(
                    transition,
                    best_cost,
                    features_proj,
                    latent_proj,
                    projector_evaluation,
                    query,
                    db.features_offset,
                    db.features_scale,
                    features_curr,
                    projector);
                
                // If projection is sufficiently different from current
                if (transition)
                {   
                    // Evaluate pose for projected features
                    decompressor_evaluate(
                        trns_bone_positions,
                        trns_bone_velocities,
                        trns_bone_rotations,
                        trns_bone_angular_velocities,
                        trns_bone_contacts,
                        decompressor_evaluation,
                        features_proj,
                        latent_proj,
                        curr_bone_positions(0),
                        curr_bone_rotations(0),
                        decompressor,
                        dt);
                    
                    // Transition inertializer to this pose
                    inertialize_pose_transition(
                        bone_offset_positions,
                        bone_offset_velocities,
                        bone_offset_rotations,
                        bone_offset_angular_velocities,
                        transition_src_position,
                        transition_src_rotation,
                        transition_dst_position,
                        transition_dst_rotation,
                        bone_positions(0),
                        bone_velocities(0),
                        bone_rotations(0),
                        bone_angular_velocities(0),
                        curr_bone_positions,
                        curr_bone_velocities,
                        curr_bone_rotations,
                        curr_bone_angular_velocities,
                        trns_bone_positions,
                        trns_bone_velocities,
                        trns_bone_rotations,
                        trns_bone_angular_velocities);
                    
                    // Update current features and latents
                    features_curr = features_proj;
                    latent_curr = latent_proj;
                }
            }
            else
            {
                // Search
                
                int best_index = end_of_anim ? -1 : frame_index;
                float best_cost = FLT_MAX;
                
                database_search(
                    best_index,
                    best_cost,
                    db,
                    query);
                selected_database_frame = best_index;
                if (logging_enabled) {
                    selected_cost = best_cost;
                    selected_terrain_error = database_raw_terrain_error(
                        db, best_index, query);
                }
                
                // Transition if better frame found
                
                if (best_index != frame_index)
                {
                    transitioned = true;
                    trns_bone_positions = db.bone_positions(best_index);
                    trns_bone_velocities = db.bone_velocities(best_index);
                    trns_bone_rotations = db.bone_rotations(best_index);
                    trns_bone_angular_velocities = db.bone_angular_velocities(best_index);
                    
                    inertialize_pose_transition(
                        bone_offset_positions,
                        bone_offset_velocities,
                        bone_offset_rotations,
                        bone_offset_angular_velocities,
                        transition_src_position,
                        transition_src_rotation,
                        transition_dst_position,
                        transition_dst_rotation,
                        bone_positions(0),
                        bone_velocities(0),
                        bone_rotations(0),
                        bone_angular_velocities(0),
                        curr_bone_positions,
                        curr_bone_velocities,
                        curr_bone_rotations,
                        curr_bone_angular_velocities,
                        trns_bone_positions,
                        trns_bone_velocities,
                        trns_bone_rotations,
                        trns_bone_angular_velocities);
                    
                    frame_index = best_index;
#ifdef MM_DISCRETE
                    dbg_did_transition = true;
                    dbg_trns_dst_rot = trns_bone_rotations(0);
#endif
                }
#ifdef MM_DISCRETE
                dbg_best_index = best_index;
#endif
            }

            // Reset search timer
            search_timer = search_time;
        }
        
        // Tick down search timer
        search_timer -= dt;

        if (lmm_enabled)
        {
            // Update features and latents
            stepper_evaluate(
                features_curr,
                latent_curr,
                stepper_evaluation,
                stepper,
                dt);
            
            // Decompress next pose
            decompressor_evaluate(
                curr_bone_positions,
                curr_bone_velocities,
                curr_bone_rotations,
                curr_bone_angular_velocities,
                curr_bone_contacts,
                decompressor_evaluation,
                features_curr,
                latent_curr,
                curr_bone_positions(0),
                curr_bone_rotations(0),
                decompressor,
                dt);
        }
        else
        {
            // Tick frame
            frame_index = database_trajectory_index_clamp(
                db, frame_index, 1);
            
            // Look-up Next Pose
            curr_bone_positions = db.bone_positions(frame_index);
            curr_bone_velocities = db.bone_velocities(frame_index);
            curr_bone_rotations = db.bone_rotations(frame_index);
            curr_bone_angular_velocities = db.bone_angular_velocities(frame_index);
            curr_bone_contacts = db.contact_states(frame_index);
        }
        
        // Update inertializer
        
        inertialize_pose_update(
            bone_positions,
            bone_velocities,
            bone_rotations,
            bone_angular_velocities,
            bone_offset_positions,
            bone_offset_velocities,
            bone_offset_rotations,
            bone_offset_angular_velocities,
            curr_bone_positions,
            curr_bone_velocities,
            curr_bone_rotations,
            curr_bone_angular_velocities,
            transition_src_position,
            transition_src_rotation,
            transition_dst_position,
            transition_dst_rotation,
            inertialize_blending_halflife,
            dt);

        motion_match_pose_diagnostic raw_selected_diagnostic;
        motion_match_pose_diagnostic inertialized_diagnostic;
        vec3 root_before_adjustment;
        if (logging_enabled) {
            array1d<vec3> raw_selected_positions(curr_bone_positions);
            array1d<quat> raw_selected_rotations(curr_bone_rotations);
            raw_selected_positions(0) = bone_positions(0);
            raw_selected_rotations(0) = bone_rotations(0);
            raw_selected_diagnostic = g1_pose_diagnostic(
                raw_selected_positions, raw_selected_rotations,
                db.bone_parents, runtime_terrain);
            inertialized_diagnostic = g1_pose_diagnostic(
                bone_positions, bone_rotations,
                db.bone_parents, runtime_terrain);
            root_before_adjustment = bone_positions(0);
        }
        
        // Update Simulation
        
        simulation_positions_update(
            simulation_position, 
            simulation_velocity, 
            simulation_acceleration,
            desired_velocity,
            simulation_velocity_halflife,
            dt);
            
        simulation_rotations_update(
            simulation_rotation, 
            simulation_angular_velocity, 
            desired_rotation,
            simulation_rotation_halflife,
            dt);
        
        // Synchronization 
        
        if (synchronization_enabled)
        {
            vec3 synchronized_position = lerp(
                simulation_position, 
                bone_positions(0),
                synchronization_data_factor);
                
            quat synchronized_rotation = quat_nlerp_shortest(
                simulation_rotation,
                bone_rotations(0), 
                synchronization_data_factor);
          
            simulation_position = synchronized_position;
            simulation_rotation = synchronized_rotation;
            
            inertialize_root_adjust(
                bone_offset_positions(0),
                transition_src_position,
                transition_src_rotation,
                transition_dst_position,
                transition_dst_rotation,
                bone_positions(0),
                bone_rotations(0),
                synchronized_position,
                synchronized_rotation);
        }
        
        // Adjustment 
        
        if (!synchronization_enabled && adjustment_enabled)
        {   
            vec3 adjusted_position = bone_positions(0);
            quat adjusted_rotation = bone_rotations(0);
            
            if (adjustment_by_velocity_enabled)
            {
                adjusted_position = adjust_character_position_by_velocity(
                    bone_positions(0),
                    bone_velocities(0),
                    simulation_position,
                    adjustment_position_max_ratio,
                    adjustment_position_halflife,
                    dt);
                
                adjusted_rotation = adjust_character_rotation_by_velocity(
                    bone_rotations(0),
                    bone_angular_velocities(0),
                    simulation_rotation,
                    adjustment_rotation_max_ratio,
                    adjustment_rotation_halflife,
                    dt);
            }
            else
            {
                adjusted_position = adjust_character_position(
                    bone_positions(0),
                    simulation_position,
                    adjustment_position_halflife,
                    dt);
                
                adjusted_rotation = adjust_character_rotation(
                    bone_rotations(0),
                    simulation_rotation,
                    adjustment_rotation_halflife,
                    dt);
            }
      
            inertialize_root_adjust(
                bone_offset_positions(0),
                transition_src_position,
                transition_src_rotation,
                transition_dst_position,
                transition_dst_rotation,
                bone_positions(0),
                bone_rotations(0),
                adjusted_position,
                adjusted_rotation);
        }

        vec3 root_after_adjustment;
        if (logging_enabled) {
            root_after_adjustment = bone_positions(0);
        }
        
        // Clamping
        
        if (!synchronization_enabled && clamping_enabled)
        {
            vec3 adjusted_position = bone_positions(0);
            quat adjusted_rotation = bone_rotations(0);
            
            adjusted_position = clamp_character_position(
                adjusted_position,
                simulation_position,
                clamping_max_distance);
            
            adjusted_rotation = clamp_character_rotation(
                adjusted_rotation,
                simulation_rotation,
                clamping_max_angle);
            
            inertialize_root_adjust(
                bone_offset_positions(0),
                transition_src_position,
                transition_src_rotation,
                transition_dst_position,
                transition_dst_rotation,
                bone_positions(0),
                bone_rotations(0),
                adjusted_position,
                adjusted_rotation);
        }

        if (logging_enabled) {
            const vec3 root_after_clamp = bone_positions(0);
            const motion_match_pose_diagnostic rendered_diagnostic =
                g1_pose_diagnostic(
                bone_positions, bone_rotations,
                db.bone_parents, runtime_terrain);
            array1d<vec3> rendered_global(db.nbones());
            array1d<quat> rendered_global_rotations(db.nbones());
            forward_kinematics_full(
                rendered_global, rendered_global_rotations,
                bone_positions, bone_rotations, db.bone_parents);
            char query_bits_hex[31 * 8 + 1] = {};
            if (!motion_match_query_bits_hex(
                    query_bits_hex, sizeof(query_bits_hex), query)) {
                fprintf(stderr,
                    "G1 runtime query error: cannot serialize finite query\n");
                controller_exit_code = 2;
                controller_exit_requested = true;
                return;
            }
            motion_match_log_row log_row;
            log_row.frame = rendered_frames;
            log_row.fixed_dt = dt;
            log_row.mode = test_config.name;
            log_row.route = test_config.route;
            log_row.query_bits_hex = query_bits_hex;
            log_row.query_database_frame = query_database_frame;
            log_row.query_range = query_range;
            log_row.selected_database_frame = selected_database_frame;
            log_row.database_frame = frame_index;
            log_row.range = g1_active_range(db, frame_index);
            log_row.source_range = g1_active_range(
                db, selected_database_frame);
            log_row.searched = search_requested;
            log_row.transitioned = transitioned;
            log_row.incumbent_cost = incumbent_cost;
            log_row.selected_cost = selected_cost;
            log_row.selected_terrain_error = selected_terrain_error;
            log_row.effective_terrain_weight =
                applied_feature_weight_terrain;
            for (int i = 0; i < 4; ++i) {
                log_row.terrain[i] = terrain_query_snapshot.values[i];
                log_row.terrain_points[i] = terrain_query_snapshot.points[i];
            }
            log_row.raw_selected = raw_selected_diagnostic;
            log_row.inertialized = inertialized_diagnostic;
            log_row.rendered = rendered_diagnostic;
            log_row.hips_inertial_offset_y =
                inertialized_diagnostic.hips_y -
                raw_selected_diagnostic.hips_y;
            log_row.runtime_root_surface_height = heightfield_sample(
                runtime_terrain, bone_positions(0).x, bone_positions(0).z);
            log_row.runtime_left_toe_surface_height = heightfield_sample(
                runtime_terrain, rendered_global(G1_LeftToe).x,
                rendered_global(G1_LeftToe).z);
            log_row.runtime_right_toe_surface_height = heightfield_sample(
                runtime_terrain, rendered_global(G1_RightToe).x,
                rendered_global(G1_RightToe).z);
            const vec3 adjustment_delta =
                root_after_adjustment - root_before_adjustment;
            const vec3 clamp_delta = root_after_clamp - root_after_adjustment;
            log_row.adjustment_xz = g1_xz_length(adjustment_delta);
            log_row.adjustment_y = adjustment_delta.y;
            log_row.clamp_xz = g1_xz_length(clamp_delta);
            log_row.clamp_y = clamp_delta.y;
            log_row.matching_enabled = matching_enabled;
            log_row.adjustment_enabled = adjustment_enabled;
            log_row.clamping_enabled = clamping_enabled;
            log_row.support_retargeting_enabled = false;
            log_row.ik_enabled = ik_enabled;
            if (!deterministic_log.write(
                    log_row, artifact_error, (int)sizeof(artifact_error))) {
                fprintf(stderr, "G1 runtime log error: %s\n", artifact_error);
                controller_exit_code = 2;
                controller_exit_requested = true;
                return;
            }
        }
        
#ifdef MM_DISCRETE
        {
            // Per-frame instrumentation. All angles in degrees.
            quat root_q   = bone_rotations(0);          // final rendered root rotation
            vec3 root_p   = bone_positions(0);
            quat off_q    = bone_offset_rotations(0);   // inertialize root ROTATION offset
            vec3 off_av   = bone_offset_angular_velocities(0);
            static float  prev_root_yaw = dbg_yaw_deg(root_q);
            static quat   prev_root_q   = root_q;
            float root_yaw   = dbg_yaw_deg(root_q);
            float jump_deg   = dbg_angle_between_deg(prev_root_q, root_q); // full 3D jump
            float des_yaw    = dbg_yaw_deg(desired_rotation);
            float sim_yaw    = dbg_yaw_deg(simulation_rotation);
            float off_ang    = dbg_quat_angle_deg(off_q);           // magnitude of root offset
            float dst_yaw    = dbg_yaw_deg(transition_dst_rotation);
            float src_yaw    = dbg_yaw_deg(transition_src_rotation);

            fprintf(g_log,
                "f=%d az=%.1f | rootYaw=%.1f jump3D=%.1f | desYaw=%.1f simYaw=%.1f "
                "| offAng=%.2f offW=%.3f offAV=%.2f | best=%d srch=%d trns=%d "
                "| dstYaw=%.1f srcYaw=%.1f | fi=%d\n",
                g_frame, camera_azimuth * 180.0f / PIf,
                root_yaw, jump_deg,
                des_yaw, sim_yaw,
                off_ang, off_q.w, length(off_av),
                dbg_best_index, (int)dbg_did_search, (int)dbg_did_transition,
                dst_yaw, src_yaw, frame_index);

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
            if (g_frame >= 400) { fflush(g_log); fclose(g_log); _Exit(0); }
        }
#endif

        // Contact fixup with foot locking and IK

        adjusted_bone_positions = bone_positions;
        adjusted_bone_rotations = bone_rotations;

        if (ik_enabled)
        {
            for (int i = 0; i < contact_bones.size; i++)
            {
                // Find all the relevant bone indices
                int toe_bone = contact_bones(i);
                int heel_bone = db.bone_parents(toe_bone);
                int knee_bone = db.bone_parents(heel_bone);
                int hip_bone = db.bone_parents(knee_bone);
                int root_bone = db.bone_parents(hip_bone);
                
                // Compute the world space position for the toe
                global_bone_computed.zero();
                
                forward_kinematics_partial(
                    global_bone_positions,
                    global_bone_rotations,
                    global_bone_computed,
                    bone_positions,
                    bone_rotations,
                    db.bone_parents,
                    toe_bone);
                
                // Update the contact state
                contact_update(
                    contact_states(i),
                    contact_locks(i),
                    contact_positions(i),  
                    contact_velocities(i),
                    contact_points(i),
                    contact_targets(i),
                    contact_offset_positions(i),
                    contact_offset_velocities(i),
                    global_bone_positions(toe_bone),
                    curr_bone_contacts(i),
                    ik_unlock_radius,
                    ik_foot_height,
                    ik_blending_halflife,
                    dt);
                
                // Ensure contact position never goes through floor
                vec3 contact_position_clamp = contact_positions(i);
                contact_position_clamp.y = maxf(contact_position_clamp.y, ik_foot_height);
                
                // Re-compute toe, heel, knee, hip, and root bone positions
                for (int bone : {heel_bone, knee_bone, hip_bone, root_bone})
                {
                    forward_kinematics_partial(
                        global_bone_positions,
                        global_bone_rotations,
                        global_bone_computed,
                        bone_positions,
                        bone_rotations,
                        db.bone_parents,
                        bone);
                }
                
                // Perform simple two-joint IK to place heel
                ik_two_bone(
                    adjusted_bone_rotations(hip_bone),
                    adjusted_bone_rotations(knee_bone),
                    global_bone_positions(hip_bone),
                    global_bone_positions(knee_bone),
                    global_bone_positions(heel_bone),
                    contact_position_clamp + (global_bone_positions(heel_bone) - global_bone_positions(toe_bone)),
                    quat_mul_vec3(global_bone_rotations(knee_bone), vec3(0.0f, 1.0f, 0.0f)),
                    global_bone_rotations(hip_bone),
                    global_bone_rotations(knee_bone),
                    global_bone_rotations(root_bone),
                    ik_max_length_buffer);
                
                // Re-compute toe, heel, and knee positions 
                global_bone_computed.zero();
                
                for (int bone : {toe_bone, heel_bone, knee_bone})
                {
                    forward_kinematics_partial(
                        global_bone_positions,
                        global_bone_rotations,
                        global_bone_computed,
                        adjusted_bone_positions,
                        adjusted_bone_rotations,
                        db.bone_parents,
                        bone);
                }
                
                // Rotate heel so toe is facing toward contact point
                ik_look_at(
                    adjusted_bone_rotations(heel_bone),
                    global_bone_rotations(knee_bone),
                    global_bone_rotations(heel_bone),
                    global_bone_positions(heel_bone),
                    global_bone_positions(toe_bone),
                    contact_position_clamp);
                
                // Re-compute toe and heel positions
                global_bone_computed.zero();
                
                for (int bone : {toe_bone, heel_bone})
                {
                    forward_kinematics_partial(
                        global_bone_positions,
                        global_bone_rotations,
                        global_bone_computed,
                        adjusted_bone_positions,
                        adjusted_bone_rotations,
                        db.bone_parents,
                        bone);
                }
                
                // Rotate toe bone so that the end of the toe 
                // does not intersect with the ground
                vec3 toe_end_curr = quat_mul_vec3(
                    global_bone_rotations(toe_bone), vec3(ik_toe_length, 0.0f, 0.0f)) + 
                    global_bone_positions(toe_bone);
                    
                vec3 toe_end_targ = toe_end_curr;
                toe_end_targ.y = maxf(toe_end_targ.y, ik_foot_height);
                
                ik_look_at(
                    adjusted_bone_rotations(toe_bone),
                    global_bone_rotations(heel_bone),
                    global_bone_rotations(toe_bone),
                    global_bone_positions(toe_bone),
                    toe_end_curr,
                    toe_end_targ);
            }
        }
        
        // Full pass of forward kinematics to compute 
        // all bone positions and rotations in the world
        // space ready for rendering
        
        forward_kinematics_full(
            global_bone_positions,
            global_bone_rotations,
            adjusted_bone_positions,
            adjusted_bone_rotations,
            db.bone_parents);
        
        // Update camera
        
        orbit_camera_update(
            camera, 
            camera_azimuth,
            camera_altitude,
            camera_distance,
            bone_positions(0) + vec3(0, 1, 0),
            // simulation_position + vec3(0, 1, 0),
            gamepadstick_right,
            desired_strafe,
            dt);

        // Render
        
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        BeginMode3D(camera);

        DrawModel(
            terrain_model,
            (Vector3){ 0.0f, 0.0f, 0.0f },
            1.0f,
            (Color){ 205, 199, 184, 255 });
        DrawModelWires(
            terrain_model,
            (Vector3){ 0.0f, 0.0f, 0.0f },
            1.0f,
            DARKGRAY);

        // Draw Simulation Object
        
        DrawCylinderWires(to_Vector3(simulation_position), 0.6f, 0.6f, 0.001f, 17, ORANGE);
        DrawSphereWires(to_Vector3(simulation_position), 0.05f, 4, 10, ORANGE);
        DrawLine3D(to_Vector3(simulation_position), to_Vector3(
            simulation_position + 0.6f * quat_mul_vec3(simulation_rotation, vec3(0.0f, 0.0f, 1.0f))), ORANGE);
        
        // Draw Clamping Radius/Angles
        
        if (clamping_enabled)
        {
            DrawCylinderWires(
                to_Vector3(simulation_position), 
                clamping_max_distance, 
                clamping_max_distance, 
                0.001f, 17, SKYBLUE);
            
            quat rotation_clamp_0 = quat_mul(quat_from_angle_axis(+clamping_max_angle, vec3(0.0f, 1.0f, 0.0f)), simulation_rotation);
            quat rotation_clamp_1 = quat_mul(quat_from_angle_axis(-clamping_max_angle, vec3(0.0f, 1.0f, 0.0f)), simulation_rotation);
            
            vec3 rotation_clamp_0_dir = simulation_position + 0.6f * quat_mul_vec3(rotation_clamp_0, vec3(0.0f, 0.0f, 1.0f));
            vec3 rotation_clamp_1_dir = simulation_position + 0.6f * quat_mul_vec3(rotation_clamp_1, vec3(0.0f, 0.0f, 1.0f));

            DrawLine3D(to_Vector3(simulation_position), to_Vector3(rotation_clamp_0_dir), SKYBLUE);
            DrawLine3D(to_Vector3(simulation_position), to_Vector3(rotation_clamp_1_dir), SKYBLUE);
        }
        
        // Draw IK foot lock positions
        
        if (ik_enabled)
        {
            for (int i = 0; i <  contact_positions.size; i++)
            {
                if (contact_locks(i))
                {
                    DrawSphereWires(to_Vector3(contact_positions(i)), 0.05f, 4, 10, PINK);
                }
            }
        }
        
        draw_trajectory(
            trajectory_positions,
            trajectory_rotations,
            ORANGE);
        
        // G1: no skinned mesh — draw the skeleton directly from bone transforms.
        // Sphere at each joint, capsule (cylinder) from each bone to its parent.
        for (int bi = 1; bi < db.nbones(); bi++)
        {
            vec3 bp = global_bone_positions(bi);
            DrawSphereWires(to_Vector3(bp), 0.028f, 4, 8, DARKBLUE);
            int par = db.bone_parents(bi);
            if (par > 0)
            {
                DrawCylinderEx(to_Vector3(global_bone_positions(par)), to_Vector3(bp),
                    0.018f, 0.018f, 6, SKYBLUE);
            }
        }
        
        // Draw matched features
        
        array1d<float> current_features = lmm_enabled ? slice1d<float>(features_curr) : db.features(frame_index);
        denormalize_features(current_features, db.features_offset, db.features_scale);        
        draw_features(current_features, bone_positions(0), bone_rotations(0), MAROON);
        
// (diagnostic MM_LOGROOT block removed)
        // Draw Simuation Bone

        DrawSphereWires(to_Vector3(bone_positions(0)), 0.05f, 4, 10, MAROON);
        DrawLine3D(to_Vector3(bone_positions(0)), to_Vector3(
            bone_positions(0) + 0.6f * quat_mul_vec3(bone_rotations(0), vec3(0.0f, 0.0f, 1.0f))), MAROON);
        
        draw_axis(vec3(), quat());

        for (int terrain_sample = 0; terrain_sample < 4; ++terrain_sample)
        {
            vec3 point = terrain_query_snapshot.points[terrain_sample];
            point.y += 0.10f;
            DrawSphereWires(to_Vector3(point), 0.04f, 4, 8, PURPLE);
        }

        EndMode3D();

        // UI
        
        //---------

        if (test_config.mode != G1_TestLive) {
            GuiDisable();
        }
        
        float ui_sim_hei = 20;
        
        GuiGroupBox((Rectangle){ 970, ui_sim_hei, 290, 250 }, "simulation object");

        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 10, 120, 20 }, 
            "velocity halflife", 
            TextFormat("%5.3f", simulation_velocity_halflife), 
            &simulation_velocity_halflife, 0.0f, 0.5f);
            
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 40, 120, 20 }, 
            "rotation halflife", 
            TextFormat("%5.3f", simulation_rotation_halflife), 
            &simulation_rotation_halflife, 0.0f, 0.5f);
            
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 70, 120, 20 }, 
            "run forward speed", 
            TextFormat("%5.3f", simulation_run_fwrd_speed), 
            &simulation_run_fwrd_speed, 0.0f, 10.0f);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 100, 120, 20 }, 
            "run sideways speed", 
            TextFormat("%5.3f", simulation_run_side_speed), 
            &simulation_run_side_speed, 0.0f, 10.0f);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 130, 120, 20 }, 
            "run backwards speed", 
            TextFormat("%5.3f", simulation_run_back_speed), 
            &simulation_run_back_speed, 0.0f, 10.0f);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 160, 120, 20 }, 
            "walk forward speed", 
            TextFormat("%5.3f", simulation_walk_fwrd_speed), 
            &simulation_walk_fwrd_speed, 0.0f, 5.0f);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 190, 120, 20 }, 
            "walk sideways speed", 
            TextFormat("%5.3f", simulation_walk_side_speed), 
            &simulation_walk_side_speed, 0.0f, 5.0f);
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_sim_hei + 220, 120, 20 }, 
            "walk backwards speed", 
            TextFormat("%5.3f", simulation_walk_back_speed), 
            &simulation_walk_back_speed, 0.0f, 5.0f);
        
        //---------
        
        float ui_inert_hei = 280;
        
        GuiGroupBox((Rectangle){ 970, ui_inert_hei, 290, 40 }, "inertiaization blending");
        
        GuiSliderBar(
            (Rectangle){ 1100, ui_inert_hei + 10, 120, 20 }, 
            "halflife", 
            TextFormat("%5.3f", inertialize_blending_halflife), 
            &inertialize_blending_halflife, 0.0f, 0.3f);
        
        //---------
        
        float ui_lmm_hei = 330;
        
        GuiGroupBox((Rectangle){ 970, ui_lmm_hei, 290, 40 }, "learned motion matching");

        GuiLabel(
            (Rectangle){ 990, ui_lmm_hei + 10, 250, 20 },
            "disabled: G1 network integration later");
        
        //---------
        
        float ui_ctrl_hei = 380;
        
        GuiGroupBox((Rectangle){ 970, ui_ctrl_hei, 290, 160 }, "controls");

        GuiLabel((Rectangle){ 990, ui_ctrl_hei +  10, 250, 20 }, "WASD / left stick - move");
        GuiLabel((Rectangle){ 990, ui_ctrl_hei +  35, 250, 20 }, "Arrows / right stick - camera");
        GuiLabel((Rectangle){ 990, ui_ctrl_hei +  60, 250, 20 }, "Left trigger - strafe");
        GuiLabel((Rectangle){ 990, ui_ctrl_hei +  85, 250, 20 }, "Shoulders - zoom");
        GuiLabel((Rectangle){ 990, ui_ctrl_hei + 110, 250, 20 }, "A button - walk");
        

        
        //---------
        
        GuiGroupBox((Rectangle){ 20, 20, 290, 260 }, "feature weights / terrain diagnostics");
        
        GuiSliderBar(
            (Rectangle){ 150, 30, 120, 20 }, 
            "foot position", 
            TextFormat("%5.3f", feature_weight_foot_position), 
            &feature_weight_foot_position, 0.001f, 3.0f);
            
        GuiSliderBar(
            (Rectangle){ 150, 60, 120, 20 }, 
            "foot velocity", 
            TextFormat("%5.3f", feature_weight_foot_velocity), 
            &feature_weight_foot_velocity, 0.001f, 3.0f);
        
        GuiSliderBar(
            (Rectangle){ 150, 90, 120, 20 }, 
            "hip velocity", 
            TextFormat("%5.3f", feature_weight_hip_velocity), 
            &feature_weight_hip_velocity, 0.001f, 3.0f);
        
        GuiSliderBar(
            (Rectangle){ 150, 120, 120, 20 }, 
            "trajectory positions", 
            TextFormat("%5.3f", feature_weight_trajectory_positions), 
            &feature_weight_trajectory_positions, 0.001f, 3.0f);
        
        GuiSliderBar(
            (Rectangle){ 150, 150, 120, 20 }, 
            "trajectory directions", 
            TextFormat("%5.3f", feature_weight_trajectory_directions), 
            &feature_weight_trajectory_directions, 0.001f, 3.0f);

        GuiSliderBar(
            (Rectangle){ 150, 180, 120, 20 },
            "terrain",
            TextFormat("%5.3f", feature_weight_terrain),
            &feature_weight_terrain, 0.0f, 10.0f);

        if (GuiButton((Rectangle){ 150, 210, 120, 20 }, "rebuild database"))
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
                feature_weight_terrain);
            if (!g1_matching_features_validate(
                    db,
                    artifact_error,
                    static_cast<int>(sizeof(artifact_error))))
            {
                fprintf(stderr, "G1 feature rebuild error: %s\n", artifact_error);
                controller_exit_code = 2;
                controller_exit_requested = true;
            }
            else
            {
                applied_feature_weight_terrain = feature_weight_terrain;
            }
        }

        GuiLabel(
            (Rectangle){ 40, 235, 250, 20 },
            TextFormat(
                "query frame %d  range %d",
                query_database_frame,
                query_range));
        GuiLabel(
            (Rectangle){ 40, 255, 250, 20 },
            TextFormat(
                "terrain %.2f %.2f %.2f %.2f",
                terrain_query_snapshot.values[0],
                terrain_query_snapshot.values[1],
                terrain_query_snapshot.values[2],
                terrain_query_snapshot.values[3]));
        
        //---------
        
        float ui_sync_hei = 290;
        
        GuiGroupBox((Rectangle){ 20, ui_sync_hei, 290, 70 }, "synchronization");

        GuiCheckBox(
            (Rectangle){ 50, ui_sync_hei + 10, 20, 20 }, 
            "enabled",
            &synchronization_enabled);

        GuiSliderBar(
            (Rectangle){ 150, ui_sync_hei + 40, 120, 20 }, 
            "data-driven amount", 
            TextFormat("%5.3f", synchronization_data_factor), 
            &synchronization_data_factor, 0.0f, 1.0f);

        //---------
        
        float ui_adj_hei = 370;
        
        GuiGroupBox((Rectangle){ 20, ui_adj_hei, 290, 130 }, "adjustment");
        
        GuiCheckBox(
            (Rectangle){ 50, ui_adj_hei + 10, 20, 20 }, 
            "enabled",
            &adjustment_enabled);    
        
        GuiCheckBox(
            (Rectangle){ 50, ui_adj_hei + 40, 20, 20 }, 
            "clamp to max velocity",
            &adjustment_by_velocity_enabled);    
        
        GuiSliderBar(
            (Rectangle){ 150, ui_adj_hei + 70, 120, 20 }, 
            "position halflife", 
            TextFormat("%5.3f", adjustment_position_halflife), 
            &adjustment_position_halflife, 0.0f, 0.5f);
        
        GuiSliderBar(
            (Rectangle){ 150, ui_adj_hei + 100, 120, 20 }, 
            "rotation halflife", 
            TextFormat("%5.3f", adjustment_rotation_halflife), 
            &adjustment_rotation_halflife, 0.0f, 0.5f);
        
        //---------
        
        float ui_clamp_hei = 510;
        
        GuiGroupBox((Rectangle){ 20, ui_clamp_hei, 290, 100 }, "clamping");
        
        GuiCheckBox(
            (Rectangle){ 50, ui_clamp_hei + 10, 20, 20 }, 
            "enabled",
            &clamping_enabled);      
        
        GuiSliderBar(
            (Rectangle){ 150, ui_clamp_hei + 40, 120, 20 }, 
            "distance", 
            TextFormat("%5.3f", clamping_max_distance), 
            &clamping_max_distance, 0.0f, 0.5f);
        
        GuiSliderBar(
            (Rectangle){ 150, ui_clamp_hei + 70, 120, 20 }, 
            "angle", 
            TextFormat("%5.3f", clamping_max_angle), 
            &clamping_max_angle, 0.0f, PIf);
        
        //---------
        
        float ui_ik_hei = 620;

        GuiGroupBox((Rectangle){ 20, ui_ik_hei, 290, 40 }, "inverse kinematics");
        GuiLabel(
            (Rectangle){ 40, ui_ik_hei + 10, 250, 20 },
            "disabled: G1 terrain IK comes later");
        
        //---------

        if (test_config.mode != G1_TestLive) {
            GuiEnable();
        }

        EndDrawing();

        ++rendered_frames;
        if (test_config.frame_limit > 0 &&
            rendered_frames >= test_config.frame_limit)
        {
            controller_exit_requested = true;
        }

    };

#if defined(PLATFORM_WEB)
    std::function<void()> u{update_func};
    emscripten_set_main_loop_arg(update_callback, &u, 0, 1);
#else
    while (!WindowShouldClose() && !controller_exit_requested)
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
#endif

    if (!deterministic_log.close(
            artifact_error, (int)sizeof(artifact_error))) {
        fprintf(stderr, "G1 runtime log error: %s\n", artifact_error);
        controller_exit_code = 2;
    }
    UnloadModel(terrain_model);

    CloseWindow();

    return controller_exit_code;
}
