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
#include "nnet.h"
#include "lmm.h"
#include "interaction_controller_adapter.h"
#include "interaction_arrival.h"
#include "interaction_debug_draw.h"
#include "interaction_pick_assist.h"
#include "interaction_pick_approach.h"
#include "interaction_runtime.h"
#include "interaction_smart_pickup_controller.h"
#include "interaction_smart_pickup_scene.h"
#include "locomotion_controller_update.h"
#include "locomotion_timing.h"
#include "stationary_motion_matching.h"

#include <algorithm>
#include <array>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

//--------------------------------------

static inline Vector3 to_Vector3(vec3 v)
{
    return (Vector3){ v.x, v.y, v.z };
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
    //   frames   0-49  : hold W (forward, -Z)
    //   frames  50-74  : hold S (backward, +Z)   <- 180 deg move reversal
    //   frames  75-99  : hold A (strafe-left, -X)
    //   frames 100-124 : hold D (strafe-right, +X)
    //   frames 125-166 : hold W again (forward)
    // Heading is ALSO snapped via camera_azimuth in the main loop.
    if (stick == GAMEPAD_STICK_LEFT)
    {
        int f = g_frame;
        if      (f < 50)  return vec3( 0.0f, 0.0f, -0.9f); // W forward
        else if (f < 75)  return vec3( 0.0f, 0.0f, +0.9f); // S backward (180 flip)
        else if (f < 100) return vec3(-0.9f, 0.0f,  0.0f); // A left
        else if (f < 125) return vec3(+0.9f, 0.0f,  0.0f); // D right
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

vec3 controller_place_staging_stick(
    const interaction::PlaceStagingPreview& preview,
    const interaction::Transform& current_root,
    float camera_azimuth)
{
    if (!preview.accepted || preview.ready)
    {
        return vec3();
    }

    vec3 root_error_world =
        preview.staging_root_world.position - current_root.position;
    root_error_world.y = 0.0F;
    vec3 world_command;
    if (preview.root_error_m > 0.02F && length(root_error_world) > 1.0e-5F)
    {
        const float position_weight = clampf(
            preview.root_error_m / 0.75F, 0.20F, 1.00F);
        world_command = position_weight * normalize(root_error_world);
    }

    if (preview.yaw_error_radians > 0.02F)
    {
        vec3 staging_facing = quat_mul_vec3(
            preview.staging_root_world.rotation,
            vec3(0.0F, 0.0F, 1.0F));
        staging_facing.y = 0.0F;
        if (length(staging_facing) > 1.0e-5F)
        {
            const float yaw_weight = 0.25F * clampf(
                preview.yaw_error_radians /
                    interaction::kPlaceStagingMaximumYawErrorRadians,
                0.0F,
                1.0F);
            world_command = world_command +
                yaw_weight * normalize(staging_facing);
        }
    }

    const float command_length = length(world_command);
    if (command_length > 1.0F)
    {
        world_command = world_command / command_length;
    }
    const quat camera_control_basis = quat_from_angle_axis(
        camera_azimuth, vec3(0.0F, 1.0F, 0.0F));
    return quat_inv_mul_vec3(camera_control_basis, world_command);
}

vec3 controller_world_navigation_stick(
    vec3 target_world,
    vec3 current_world,
    float camera_azimuth,
    float slow_radius_m)
{
    vec3 world_command = target_world - current_world;
    world_command.y = 0.0F;
    const float distance_m = length(world_command);
    if (distance_m <= 1.0e-5F)
    {
        return vec3();
    }
    const float magnitude = slow_radius_m > 0.0F
        ? clampf(distance_m / slow_radius_m, 0.15F, 1.0F)
        : 1.0F;
    world_command = magnitude * normalize(world_command);
    const quat camera_control_basis = quat_from_angle_axis(
        camera_azimuth, vec3(0.0F, 1.0F, 0.0F));
    return quat_inv_mul_vec3(camera_control_basis, world_command);
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
  const float back_speed)
{
    desired_velocities(0) = desired_velocity;
    
    for (int i = 1; i < desired_velocities.size; i++)
    {
        desired_velocities(i) = desired_velocity_update(
            gamepadstick_left,
            orbit_camera_update_azimuth(
                camera_azimuth,
                gamepadstick_right,
                desired_strafe,
                locomotion_timing::kTrajectorySampleTimesSeconds[
                    static_cast<size_t>(i - 1)]),
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
    const slice1d<vec3> obstacles_positions,
    const slice1d<vec3> obstacles_scales)
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
            locomotion_timing::kTrajectoryStepSeconds[
                static_cast<size_t>(i - 1)],
            obstacles_positions, 
            obstacles_scales);
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
  const bool desired_strafe)
{
    desired_rotations(0) = desired_rotation;
    
    for (int i = 1; i < desired_rotations.size; i++)
    {
        desired_rotations(i) = desired_rotation_update(
            desired_rotations(i-1),
            gamepadstick_left,
            gamepadstick_right,
            orbit_camera_update_azimuth(
                camera_azimuth,
                gamepadstick_right,
                desired_strafe,
                locomotion_timing::kTrajectorySampleTimesSeconds[
                    static_cast<size_t>(i - 1)]),
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
    const float halflife)
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
            locomotion_timing::kTrajectorySampleTimesSeconds[
                static_cast<size_t>(i - 1)]);
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

void draw_obstacles(
    const slice1d<vec3> obstacles_positions,
    const slice1d<vec3> obstacles_scales)
{
    for (int i = 0; i < obstacles_positions.size; i++)
    {
        vec3 position = vec3(
            obstacles_positions(i).x, 
            obstacles_positions(i).y + 0.5f * obstacles_scales(i).y + 0.01f, 
            obstacles_positions(i).z);
      
        DrawCube(
            to_Vector3(position),
            obstacles_scales(i).x, 
            obstacles_scales(i).y, 
            obstacles_scales(i).z,
            LIGHTGRAY);
            
        DrawCubeWires(
            to_Vector3(position),
            obstacles_scales(i).x, 
            obstacles_scales(i).y, 
            obstacles_scales(i).z,
            GRAY);
    }
}

//--------------------------------------

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

namespace {

bool autodemo_is_finite(float value)
{
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

void save_matching_features_checked(
    const database& db,
    const std::filesystem::path& path)
{
    if (db.features.rows < 0 || db.features.cols < 0 ||
        db.features_offset.size < 0 || db.features_scale.size < 0)
    {
        throw std::runtime_error(
            "matching features contain invalid dimensions");
    }

    const std::string filename = path.string();
    FILE* output = std::fopen(filename.c_str(), "wb");
    if (output == nullptr)
    {
        throw std::runtime_error(
            "cannot open matching features output " + filename);
    }

    const size_t feature_value_count =
        static_cast<size_t>(db.features.rows) *
        static_cast<size_t>(db.features.cols);
    bool write_succeeded = true;
    const auto write_exact =
        [&](const void* values, size_t element_size, size_t count)
        {
            if (std::fwrite(values, element_size, count, output) != count)
            {
                write_succeeded = false;
            }
        };
    write_exact(&db.features.rows, sizeof(int), 1U);
    write_exact(&db.features.cols, sizeof(int), 1U);
    write_exact(
        db.features.data,
        sizeof(float),
        feature_value_count);
    write_exact(&db.features_offset.size, sizeof(int), 1U);
    write_exact(
        db.features_offset.data,
        sizeof(float),
        static_cast<size_t>(db.features_offset.size));
    write_exact(&db.features_scale.size, sizeof(int), 1U);
    write_exact(
        db.features_scale.data,
        sizeof(float),
        static_cast<size_t>(db.features_scale.size));

    const int flush_result = std::fflush(output);
    const int close_result = std::fclose(output);
    if (!write_succeeded || flush_result != 0 || close_result != 0)
    {
        throw std::runtime_error(
            "cannot write matching features output " + filename);
    }
}

enum class AutodemoMode
{
    Pickup,
    Placement,
};

struct AutodemoConfiguration
{
    AutodemoMode mode = AutodemoMode::Pickup;
    std::filesystem::path log_final;
    std::filesystem::path log_temporary;
    std::filesystem::path log_backup;
    std::filesystem::path screenshot_final;
    std::filesystem::path screenshot_temporary;
    std::filesystem::path screenshot_backup;
    bool preserve_backups_after_rollback_failure = false;
};

struct AutodemoEvidenceCapture
{
    bool grasp_evidence_valid = false;
    int active_hand_joint = -1;
    float hand_constraint_weight = 0.0F;
    bool hand_constraint_validated = false;
    interaction::TargetRigArmIKResult hand_constraint_result{};
    quat hand_constraint_calibration_rotation{};
    quat calibrated_hand_world_rotation{};
    interaction::Transform object_world{};
    interaction::Transform hand_in_object{};
    interaction::Transform grasp_world{};
    std::array<vec3, interaction::kFlatControllerBoneCount> joint_positions{};
    std::array<quat, interaction::kFlatControllerBoneCount> joint_rotations{};
};

bool autodemo_hand_constraint_evidence_valid(
    const AutodemoEvidenceCapture& capture)
{
    return autodemo_is_finite(capture.hand_constraint_weight) &&
        capture.hand_constraint_weight >= 0.0F &&
        capture.hand_constraint_weight <= 1.0F &&
        (capture.hand_constraint_weight == 0.0F ||
         (capture.hand_constraint_validated &&
          capture.hand_constraint_result.applied &&
          capture.hand_constraint_result.reachable));
}

constexpr std::array<const char*, interaction::kFlatControllerBoneCount>
    kAutodemoFlatJointNames = {
        "Entity",
        "Hips",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToe",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToe",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand"};

AutodemoEvidenceCapture capture_autodemo_evidence(
    const slice1d<vec3> global_bone_positions,
    const slice1d<quat> global_bone_rotations,
    const interaction::RuntimeOutput& runtime_output,
    const interaction::ControllerInteractionFrameState& frame_state,
    const interaction::InteractionTarget* scene_target,
    const interaction::Transform& object_world)
{
    if (global_bone_positions.size !=
            static_cast<int>(interaction::kFlatControllerBoneCount) ||
        global_bone_rotations.size !=
            static_cast<int>(interaction::kFlatControllerBoneCount))
    {
        throw std::runtime_error(
            "autodemo rendered joint arrays must contain exactly 23 joints");
    }

    AutodemoEvidenceCapture capture;
    capture.hand_constraint_weight =
        runtime_output.diagnostics.hand_constraint_weight;
    capture.hand_constraint_validated =
        frame_state.hand_constraint_validated;
    capture.hand_constraint_result = frame_state.hand_constraint_result;
    capture.hand_constraint_calibration_rotation = quat_normalize(
        frame_state.hand_constraint_calibration_rotation);
    capture.object_world = object_world;
    for (size_t joint = 0;
         joint < interaction::kFlatControllerBoneCount;
         ++joint)
    {
        const int index = static_cast<int>(joint);
        capture.joint_positions[joint] = global_bone_positions(index);
        capture.joint_rotations[joint] =
            quat_normalize(global_bone_rotations(index));
    }

    const interaction::RuntimeState state =
        runtime_output.diagnostics.state;
    const bool interaction_owned_state =
        state == interaction::RuntimeState::PickupReplay ||
        state == interaction::RuntimeState::Hold ||
        state == interaction::RuntimeState::Carry ||
        state == interaction::RuntimeState::PlacePreflight ||
        state == interaction::RuntimeState::PlaceAlign ||
        state == interaction::RuntimeState::PlaceReplay;
    const bool selected_target_agrees =
        scene_target != nullptr &&
        runtime_output.diagnostics.target == scene_target->handle;
    if (!interaction_owned_state || !runtime_output.owns_pose ||
        !selected_target_agrees)
    {
        return capture;
    }

    for (const interaction::GraspAffordance& affordance :
         scene_target->affordances)
    {
        if (affordance.id == runtime_output.diagnostics.affordance_id &&
            affordance.hand == runtime_output.diagnostics.hand)
        {
            capture.grasp_evidence_valid = true;
            capture.active_hand_joint =
                affordance.hand == interaction::Hand::Left ? 18 : 22;
            capture.hand_in_object = affordance.hand_in_object;
            capture.grasp_world = interaction::compose(capture.object_world, capture.hand_in_object);
            if (capture.hand_constraint_validated)
            {
                capture.calibrated_hand_world_rotation = quat_normalize(quat_mul(
                    capture.joint_rotations[static_cast<size_t>(capture.active_hand_joint)],
                    quat_inv(capture.hand_constraint_calibration_rotation)
                ));
            }
            break;
        }
    }
    return capture;
}

std::filesystem::path autodemo_normalized_path(
    const std::filesystem::path& path)
{
    return std::filesystem::absolute(path).lexically_normal();
}

void autodemo_require_parent_directory(
    const std::filesystem::path& path,
    const char* label)
{
    const std::filesystem::path parent = path.parent_path().empty()
        ? std::filesystem::path(".")
        : path.parent_path();
    std::error_code error;
    const bool directory = std::filesystem::is_directory(parent, error);
    if (error || !directory)
    {
        throw std::runtime_error(
            std::string(label) + " parent directory does not exist");
    }
}

std::optional<AutodemoConfiguration> parse_autodemo_environment()
{
    const char* autodemo_value = std::getenv("MM_INTERACTION_AUTODEMO");
    const char* placement_autodemo_value =
        std::getenv("MM_INTERACTION_PLACE_AUTODEMO");
    if (autodemo_value == nullptr && placement_autodemo_value == nullptr)
    {
        return std::nullopt;
    }
    if (autodemo_value != nullptr && placement_autodemo_value != nullptr)
    {
        throw std::runtime_error(
            "pickup and placement auto-demo modes are mutually exclusive");
    }
    if (autodemo_value != nullptr && std::string(autodemo_value) != "1")
    {
        throw std::runtime_error("MM_INTERACTION_AUTODEMO must equal 1");
    }
    if (placement_autodemo_value != nullptr &&
        std::string(placement_autodemo_value) != "1")
    {
        throw std::runtime_error(
            "MM_INTERACTION_PLACE_AUTODEMO must equal 1");
    }

    const char* log_value = std::getenv("MM_INTERACTION_LOG");
    const char* screenshot_value =
        std::getenv("MM_INTERACTION_SCREENSHOT");
    if (log_value == nullptr || log_value[0] == '\0')
    {
        throw std::runtime_error("MM_INTERACTION_LOG must be nonempty");
    }
    if (screenshot_value == nullptr || screenshot_value[0] == '\0')
    {
        throw std::runtime_error(
            "MM_INTERACTION_SCREENSHOT must be nonempty");
    }

    AutodemoConfiguration configuration;
    configuration.mode = placement_autodemo_value != nullptr
        ? AutodemoMode::Placement
        : AutodemoMode::Pickup;
    configuration.log_final = std::filesystem::path(log_value);
    configuration.screenshot_final =
        std::filesystem::path(screenshot_value);
    autodemo_require_parent_directory(configuration.log_final, "log");
    autodemo_require_parent_directory(
        configuration.screenshot_final, "screenshot");
    if (autodemo_normalized_path(configuration.log_final) ==
        autodemo_normalized_path(configuration.screenshot_final))
    {
        throw std::runtime_error(
            "MM_INTERACTION_LOG and MM_INTERACTION_SCREENSHOT must differ");
    }

    configuration.log_temporary = std::filesystem::path(
        configuration.log_final.string() + ".tmp");
    configuration.log_backup = std::filesystem::path(
        configuration.log_final.string() + ".autodemo-previous");
    configuration.screenshot_temporary =
        configuration.screenshot_final.parent_path() /
        (configuration.screenshot_final.filename().string() + ".tmp.png");
    configuration.screenshot_backup = std::filesystem::path(
        configuration.screenshot_final.string() + ".autodemo-previous");

    const std::array<std::filesystem::path, 6> paths = {
        configuration.log_final,
        configuration.log_temporary,
        configuration.log_backup,
        configuration.screenshot_final,
        configuration.screenshot_temporary,
        configuration.screenshot_backup};
    for (size_t left = 0; left < paths.size(); ++left)
    {
        for (size_t right = left + 1; right < paths.size(); ++right)
        {
            if (autodemo_normalized_path(paths[left]) ==
                autodemo_normalized_path(paths[right]))
            {
                throw std::runtime_error(
                    "autodemo final and temporary paths must be distinct");
            }
        }
    }
    return configuration;
}

void autodemo_remove_checked(const std::filesystem::path& path)
{
    std::error_code error;
    (void)std::filesystem::remove(path, error);
    if (error)
    {
        throw std::runtime_error(
            "cannot remove autodemo temporary " + path.string());
    }
}

void prepare_autodemo_temporaries(
    const AutodemoConfiguration& configuration)
{
    autodemo_remove_checked(configuration.log_temporary);
    autodemo_remove_checked(configuration.log_backup);
    autodemo_remove_checked(configuration.screenshot_temporary);
    autodemo_remove_checked(configuration.screenshot_backup);
}

void cleanup_autodemo_backups_best_effort(
    const AutodemoConfiguration& configuration) noexcept
{
    const std::array<const std::filesystem::path*, 2> backup_paths = {
        &configuration.log_backup,
        &configuration.screenshot_backup};
    for (const std::filesystem::path* path : backup_paths)
    {
        std::error_code ignored;
        (void)std::filesystem::remove(*path, ignored);
    }
}

void cleanup_autodemo_temporaries(
    const AutodemoConfiguration& configuration) noexcept
{
    const std::array<std::filesystem::path, 2> temporary_paths = {
        configuration.log_temporary,
        configuration.screenshot_temporary};
    for (const std::filesystem::path& path : temporary_paths)
    {
        std::error_code ignored;
        (void)std::filesystem::remove(path, ignored);
    }
    if (!configuration.preserve_backups_after_rollback_failure)
    {
        cleanup_autodemo_backups_best_effort(configuration);
    }
}

void cleanup_autodemo_temporaries_checked(
    const AutodemoConfiguration& configuration)
{
    autodemo_remove_checked(configuration.log_temporary);
    autodemo_remove_checked(configuration.screenshot_temporary);
}

struct AutodemoPublicationSlot
{
    std::filesystem::path temporary;
    std::filesystem::path final;
    std::filesystem::path backup;
    bool final_existed = false;
    bool published = false;
};

void prepare_autodemo_backup(AutodemoPublicationSlot& slot)
{
    std::error_code error;
    slot.final_existed = std::filesystem::exists(slot.final, error);
    if (error)
    {
        throw std::runtime_error(
            "cannot inspect prior autodemo evidence " +
            slot.final.string());
    }
    if (!slot.final_existed)
    {
        return;
    }
    const bool copied = std::filesystem::copy_file(
        slot.final,
        slot.backup,
        std::filesystem::copy_options::overwrite_existing,
        error);
    if (error || !copied)
    {
        throw std::runtime_error(
            "cannot preserve prior autodemo evidence " +
            slot.final.string());
    }
}

void publish_autodemo_slot(AutodemoPublicationSlot& slot)
{
    std::error_code error;
    std::filesystem::rename(slot.temporary, slot.final, error);
    if (error)
    {
        throw std::runtime_error(
            "cannot publish autodemo evidence " + slot.final.string());
    }
    slot.published = true;
}

bool rollback_autodemo_slot(AutodemoPublicationSlot& slot) noexcept
{
    if (!slot.published)
    {
        return true;
    }
    std::error_code error;
    if (slot.final_existed)
    {
        std::filesystem::rename(slot.backup, slot.final, error);
    }
    else
    {
        (void)std::filesystem::remove(slot.final, error);
    }
    slot.published = false;
    return !error;
}

void publish_autodemo_evidence(
    AutodemoConfiguration& configuration)
{
    AutodemoPublicationSlot screenshot{
        configuration.screenshot_temporary,
        configuration.screenshot_final,
        configuration.screenshot_backup};
    AutodemoPublicationSlot log{
        configuration.log_temporary,
        configuration.log_final,
        configuration.log_backup};
    try
    {
        prepare_autodemo_backup(screenshot);
        prepare_autodemo_backup(log);
        publish_autodemo_slot(screenshot);
        publish_autodemo_slot(log);
    }
    catch (...)
    {
        const bool log_restored = rollback_autodemo_slot(log);
        const bool screenshot_restored = rollback_autodemo_slot(screenshot);
        if (!log_restored || !screenshot_restored)
        {
            configuration.preserve_backups_after_rollback_failure = true;
            throw std::runtime_error(
                "autodemo publication failed and rollback was incomplete");
        }
        cleanup_autodemo_temporaries(configuration);
        throw;
    }
    cleanup_autodemo_temporaries_checked(configuration);
    cleanup_autodemo_backups_best_effort(configuration);
}

vec3 autodemo_read_vec3(
    const std::vector<float>& values,
    size_t index)
{
    const size_t offset = index * 3U;
    return vec3(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U));
}

quat autodemo_read_quat(
    const std::vector<float>& values,
    size_t index)
{
    const size_t offset = index * 4U;
    return quat(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U),
        values.at(offset + 3U));
}

float autodemo_yaw_radians(quat rotation)
{
    const vec3 facing =
        quat_mul_vec3(rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(facing.x, facing.z);
}

float autodemo_shortest_angle(float angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

interaction::Transform autodemo_scene_alignment(
    const interaction::Transform& source_object,
    const interaction::Transform& target_object)
{
    const float yaw = autodemo_shortest_angle(
        autodemo_yaw_radians(target_object.rotation) -
        autodemo_yaw_radians(source_object.rotation));
    const quat rotation =
        quat_from_angle_axis(yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source =
        quat_mul_vec3(rotation, source_object.position);
    return {
        vec3(
            target_object.position.x - rotated_source.x,
            0.0F,
            target_object.position.z - rotated_source.z),
        rotation};
}

interaction::Pose autodemo_mapped_pose(
    const interaction::Database& database,
    int32_t frame,
    const interaction::Transform& scene_from_source)
{
    interaction::Pose pose = interaction::pose_at_frame(database, frame);
    constexpr size_t root = g1_skeleton::Simulation;
    const interaction::Transform mapped_root = interaction::compose(
        scene_from_source,
        {pose.positions[root], pose.rotations[root]});
    pose.positions[root] = mapped_root.position;
    pose.rotations[root] = mapped_root.rotation;
    pose.velocities[root] = quat_mul_vec3(
        scene_from_source.rotation, pose.velocities[root]);
    pose.angular_velocities[root] = quat_mul_vec3(
        scene_from_source.rotation, pose.angular_velocities[root]);
    return pose;
}

struct AutodemoCanonicalEntry
{
    interaction::LocomotionSnapshot snapshot{};
    std::array<vec3, 3> future_root_velocities{};
    std::array<vec3, 3> future_root_angular_velocities{};
};

AutodemoCanonicalEntry make_autodemo_canonical_entry(
    const interaction::Database& database,
    const interaction::Features& features,
    const interaction::InteractionTarget& target)
{
    if (database.clip_count == 0U || database.range_starts.empty() ||
        database.range_stops.empty())
    {
        throw std::runtime_error("autodemo pack has no clip 0");
    }
    if (target.affordances.size() != 1U)
    {
        throw std::runtime_error(
            "autodemo target must have exactly one affordance");
    }

    const int32_t start = database.range_starts.at(0U);
    const int32_t stop = database.range_stops.at(0U);
    if (start < 0 || stop <= start ||
        stop > static_cast<int32_t>(database.frame_count))
    {
        throw std::runtime_error("autodemo clip 0 range is invalid");
    }

    int32_t reach = -1;
    int32_t contact = -1;
    for (int32_t frame = start; frame < stop; ++frame)
    {
        const uint8_t phase =
            database.phases.at(static_cast<size_t>(frame));
        if (reach < 0 &&
            phase == static_cast<uint8_t>(interaction::Phase::Reach))
        {
            reach = frame;
        }
        if (contact < 0 &&
            phase == static_cast<uint8_t>(interaction::Phase::Contact))
        {
            contact = frame;
        }
    }
    if (reach < start || contact <= reach || contact - 1 < start)
    {
        throw std::runtime_error(
            "autodemo clip 0 is missing ordered Reach/Contact phases");
    }

    const interaction::Transform source_object{
        autodemo_read_vec3(
            database.object_positions,
            static_cast<size_t>(contact - 1)),
        autodemo_read_quat(
            database.object_rotations,
            static_cast<size_t>(contact - 1))};
    const interaction::Transform scene_from_source =
        autodemo_scene_alignment(source_object, target.object_world);

    AutodemoCanonicalEntry entry;
    entry.snapshot.pose =
        autodemo_mapped_pose(database, reach, scene_from_source);
    constexpr std::array<int32_t, 3> future_offsets = {8, 17, 25};
    for (size_t index = 0; index < future_offsets.size(); ++index)
    {
        const int32_t future_frame =
            std::min(reach + future_offsets[index], stop - 1);
        const interaction::Pose future =
            autodemo_mapped_pose(database, future_frame, scene_from_source);
        constexpr size_t root = g1_skeleton::Simulation;
        entry.snapshot.future_root_positions[index] =
            future.positions[root];
        entry.snapshot.future_root_rotations[index] =
            future.rotations[root];
        entry.future_root_velocities[index] = future.velocities[root];
        entry.future_root_angular_velocities[index] =
            future.angular_velocities[root];
    }

    const interaction::GraspAffordance& affordance =
        target.affordances.front();
    interaction::QueryInput query_input{};
    query_input.locomotion = entry.snapshot;
    query_input.grasp_world =
        interaction::compose(target.object_world, affordance.hand_in_object);
    query_input.table_world = target.table_world;
    query_input.table_size = target.table_size;
    query_input.approach_direction_object =
        affordance.approach_direction_object;
    query_input.object_dimensions = target.object_dimensions;
    query_input.hand = affordance.hand;
    const interaction::NormalizedQuery query = interaction::normalize_query(
        interaction::build_raw_query(query_input), features);
    if (features.dimension != interaction::kFeatureDimension ||
        features.feature_count != interaction::kFeatureDimension ||
        features.frame_count <= static_cast<uint32_t>(reach) ||
        features.values.size() <
            static_cast<size_t>(features.frame_count) * features.dimension)
    {
        throw std::runtime_error(
            "autodemo interaction feature row is unavailable");
    }
    float maximum_error = 0.0F;
    constexpr size_t kAutodemoLocomotionFeatureStop = 45U;
    for (size_t dimension = 0;
         dimension < kAutodemoLocomotionFeatureStop;
         ++dimension)
    {
        const float expected = features.values.at(
            static_cast<size_t>(reach) * features.dimension + dimension);
        if (!autodemo_is_finite(query[dimension]) ||
            !autodemo_is_finite(expected))
        {
            throw std::runtime_error(
                "autodemo canonical locomotion query is non-finite");
        }
        maximum_error = std::max(
            maximum_error, std::abs(query[dimension] - expected));
    }
    if (maximum_error > 2.0e-4F)
    {
        throw std::runtime_error(
            "autodemo canonical locomotion query differs from clip-0 Reach row");
    }
    return entry;
}

enum class AutodemoAction
{
    None,
    Approach,
    Interact,
    Forward,
    LatchDestination,
    Stage,
    Place,
    Reset,
};

volatile std::sig_atomic_t autodemo_sigterm_requested = 0;

void autodemo_sigterm_handler(int) noexcept
{
    autodemo_sigterm_requested = 1;
}

const char* autodemo_action_name(AutodemoAction action)
{
    switch (action)
    {
    case AutodemoAction::None: return "none";
    case AutodemoAction::Approach: return "approach";
    case AutodemoAction::Interact: return "interact";
    case AutodemoAction::Forward: return "forward";
    case AutodemoAction::LatchDestination: return "latch_destination";
    case AutodemoAction::Stage: return "stage";
    case AutodemoAction::Place: return "place";
    case AutodemoAction::Reset: return "reset";
    }
    return "none";
}

constexpr uint32_t kAutodemoResetPresentationFrames = 7U;
constexpr uint32_t kAutodemoWarmupFrames = 50U;
constexpr uint64_t kAutodemoInteractFrame = 13U;
constexpr int kAutodemoCarryCommandCount = 63;
constexpr int kAutodemoFinalCarryCommand =
    kAutodemoCarryCommandCount - 1;
constexpr uint64_t kAutodemoCarryDeadlineFrames = 375U;
constexpr uint64_t kAutodemoMaximumEvidenceFrames = 500U;
static_assert(
    kAutodemoResetPresentationFrames ==
        locomotion_timing::ticks_for_milliseconds(250U));
static_assert(
    kAutodemoWarmupFrames ==
        locomotion_timing::ticks_for_milliseconds(2000U));
static_assert(
    kAutodemoInteractFrame ==
        locomotion_timing::ticks_for_milliseconds(500U));
static_assert(
    kAutodemoCarryCommandCount == static_cast<int>(
        locomotion_timing::ticks_for_milliseconds(2500U)));

struct ControllerAutodemoState
{
    std::ofstream log;
    uint64_t render_frame = 0U;
    uint64_t runtime_tick = 0U;
    uint32_t warmup_render_ticks = 0U;
    uint32_t reset_presentation_frames_remaining = 0U;
    int carry_command_count = 0;
    bool evidence_started = false;
    bool interact_pulsed = false;
    bool candidate_verified = false;
    bool carry_origin_captured = false;
    bool reset_pending = false;
    bool screenshot_captured = false;
    bool complete = false;
    bool exit_requested = false;
    vec3 carry_origin{};
    float last_carry_displacement_m = 0.0F;
    std::vector<interaction::RuntimeState> collapsed_states;
    std::string failure;
};

constexpr uint64_t kPlacementAutodemoMaximumEvidenceFrames = 900U;
constexpr uint32_t kPlacementAutodemoMinimumWalkTicks = 25U;
constexpr uint32_t kPlacementAutodemoMaximumWalkTicks = 250U;
constexpr uint32_t kPlacementAutodemoSettleTicks = 5U;
constexpr float kPlacementAutodemoInitialDistanceM = 2.80F;
constexpr float kPlacementAutodemoMinimumWalkDisplacementM = 2.00F;
constexpr float kPlacementAutodemoSettleMaximumSpeedMps = 0.10F;
constexpr float kPlacementAutodemoReachEntryDistanceM = 0.60F;
constexpr float kPlacementAutodemoReachEntryToleranceM = 0.12F;
constexpr float kPlacementAutodemoAlignmentPositionErrorM = 0.15F;
constexpr interaction::ArrivalConfig kPlacementAutodemoArrivalConfig{};
constexpr float kPlacementAutodemoReachYawErrorRadians =
    20.0F * PIf / 180.0F;
constexpr float kPlacementAutodemoNavigationSlowRadiusM = 0.35F;
constexpr uint32_t kPlacementAutodemoMinimumCarryTicks = 25U;
constexpr uint32_t kPlacementAutodemoMaximumCarryTicks = 150U;
constexpr uint32_t kPlacementAutodemoPickupToCarryTicks = 375U;
constexpr uint32_t kPlacementAutodemoPlaceTicks = 250U;
// Keep the placement demo alive through the complete 0.50 s post-release
// arm return, then require three frames fully relinquished to ordinary
// locomotion before publication. "Settled" here means no interaction pose
// overlay; the locomotion database may still contain natural idle motion.
// This is intentionally separate from pickup's seven-frame, unlogged Reset
// presentation drain above.
constexpr uint32_t kPlacementAutodemoReturnFrames = 15U;
constexpr uint32_t kPlacementAutodemoSettledFrames = 3U;
constexpr uint32_t kPlacementAutodemoHandoffFrames =
    kPlacementAutodemoReturnFrames +
    kPlacementAutodemoSettledFrames;
constexpr uint32_t kPlacementAutodemoMaximumHandoffFrames = 75U;
static_assert(
    kPlacementAutodemoReturnFrames ==
        locomotion_timing::ticks_for_milliseconds(600U));
static_assert(
    kPlacementAutodemoSettledFrames ==
        locomotion_timing::ticks_for_milliseconds(120U));
static_assert(kPlacementAutodemoHandoffFrames == 18U);
static_assert(
    kPlacementAutodemoMaximumHandoffFrames ==
        locomotion_timing::ticks_for_milliseconds(3000U));

enum class PlacementPickPreviewFailure : uint8_t
{
    None,
    NoPath,
    Deadline,
};

const char* placement_pick_preview_failure_name(
    PlacementPickPreviewFailure failure)
{
    switch (failure)
    {
    case PlacementPickPreviewFailure::None: return "none";
    case PlacementPickPreviewFailure::NoPath: return "no_path";
    case PlacementPickPreviewFailure::Deadline: return "deadline";
    }
    throw std::runtime_error(
        "placement pickup preview has an invalid pending failure");
}

const char* placement_pick_preview_reason_name(interaction::Reason reason)
{
    switch (reason)
    {
    case interaction::Reason::None: return "None";
    case interaction::Reason::PackUnavailable: return "PackUnavailable";
    case interaction::Reason::TargetUnavailable: return "TargetUnavailable";
    case interaction::Reason::TargetChanged: return "TargetChanged";
    case interaction::Reason::OutOfRange: return "OutOfRange";
    case interaction::Reason::NoCandidate: return "NoCandidate";
    case interaction::Reason::PoorMatch: return "PoorMatch";
    case interaction::Reason::BlockedPath: return "BlockedPath";
    case interaction::Reason::CorrectionLimit: return "CorrectionLimit";
    case interaction::Reason::Cancelled: return "Cancelled";
    case interaction::Reason::ContactPosition: return "ContactPosition";
    case interaction::Reason::ContactOrientation:
        return "ContactOrientation";
    case interaction::Reason::JointLimit: return "JointLimit";
    case interaction::Reason::LostContact: return "LostContact";
    case interaction::Reason::ClipEnded: return "ClipEnded";
    case interaction::Reason::Reset: return "Reset";
    case interaction::Reason::SurfaceUnavailable: return "SurfaceUnavailable";
    case interaction::Reason::SurfaceChanged: return "SurfaceChanged";
    case interaction::Reason::PlacementOutOfBounds:
        return "PlacementOutOfBounds";
    case interaction::Reason::ReleasePosition: return "ReleasePosition";
    case interaction::Reason::ReleaseOrientation: return "ReleaseOrientation";
    }
    throw std::runtime_error(
        "placement pickup preview has an invalid reason");
}

struct ControllerPlacementAutodemoState
{
    interaction::Transform reach_waypoint{};
    interaction::PickEntrySlots pick_entry_slots{};
    std::array<interaction::PickEntryPreview, 2> pick_entry_previews{};
    interaction::Transform interaction_waypoint{};
    vec3 reach_entry_point{};
    std::optional<interaction::PlaceStagingPreview> live_preview{};
    interaction::PlacementFit final_actual_fit{};
    vec3 walking_origin{};
    vec3 previous_displayed_root{};
    vec3 carry_origin_root{};
    vec3 carry_origin_object{};
    vec3 left_stick_command{};
    uint64_t render_frame = 0U;
    int64_t pick_preview_epoch = -1;
    int64_t pick_preview_consumed_epoch = -1;
    uint64_t pick_preview_epoch_count = 0U;
    uint64_t pick_preview_snapshot_fingerprint = 0U;
    uint32_t walk_ticks = 0U;
    uint32_t pick_preview_elapsed_ticks = 0U;
    uint32_t settle_ticks = 0U;
    uint32_t pickup_to_carry_ticks = 0U;
    uint32_t carry_staging_ticks = 0U;
    uint32_t place_ticks = 0U;
    uint32_t post_entry_ticks = 0U;
    uint32_t handoff_frames_remaining = 0U;
    uint32_t handoff_frames_recorded = 0U;
    uint32_t post_release_settled_frames_recorded = 0U;
    uint64_t runtime_preview_calls = 0U;
    uint64_t runtime_pick_preview_calls = 0U;
    uint64_t pick_preview_mutation_count = 0U;
    uint64_t locomotion_provider_calls = 0U;
    uint64_t live_flat_provider_calls = 0U;
    uint64_t canonical_snapshot_provider_calls = 0U;
    uint64_t root_relocation_calls = 0U;
    uint64_t simulation_root_initialization_calls = 0U;
    uint64_t displayed_root_initialization_calls = 0U;
    uint64_t pick_resolver_calls = 0U;
    uint64_t placement_surface_resolver_calls = 0U;
    uint64_t free_selector_calls = 0U;
    uint64_t external_match_input_calls = 0U;
    uint64_t preview_mutation_count = 0U;
    uint64_t direct_root_write_calls = 0U;
    uint64_t initial_far_selection_id = 0U;
    uint64_t current_selection_id = 0U;
    uint64_t submitted_selection_id = 0U;
    uint64_t preview_ik_fingerprint = 0U;
    uint64_t pick_interact_submission_count = 0U;
    uint64_t pick_reservation_transition_count = 0U;
    uint32_t attachment_transition_count = 0U;
    uint32_t attached_to_free_transition_count = 0U;
    uint64_t stationary_candidate_count = 0U;
    uint64_t stationary_search_calls = 0U;
    uint64_t stationary_transition_count = 0U;
    int stationary_selected_frame = -1;
    int stationary_selected_range = -1;
    float initial_pickup_distance_m = 0.0F;
    float last_root_speed_mps = 0.0F;
    float interaction_lateral_offset_m = 0.0F;
    float brake_simulation_speed_mps = 0.0F;
    bool origin_captured = false;
    bool evidence_started = false;
    bool interact_pulsed = false;
    bool reach_entry_reached = false;
    bool reach_braking = false;
    bool stationary_constraint_active = false;
    bool destination_latched = false;
    bool last_provider_was_live_flat = false;
    bool pick_resolver_used_live_flat = false;
    bool destination_generation_changed_before_place = false;
    bool initial_preview_verified = false;
    bool place_observed = false;
    bool actual_fit_captured = false;
    bool final_support_sweep_clear = false;
    bool destination_support_committed = false;
    bool attachment_initialized = false;
    bool last_attached = false;
    std::optional<size_t> frozen_pick_entry_index{};
    std::optional<uint64_t> pick_entry_freeze_tick{};
    PlacementPickPreviewFailure pending_pick_preview_failure =
        PlacementPickPreviewFailure::None;
    bool pick_preview_failure_recorded = false;
    bool final_record_written = false;
    bool screenshot_captured = false;
    std::vector<interaction::RuntimeState> collapsed_states;
};

struct ManualPickStationaryDiagnostics
{
    uint64_t search_count = 0U;
    uint64_t transition_count = 0U;
    int selected_frame = -1;
    float selected_cost = 0.0F;
};

struct PlacementPickPreviewObservation
{
    interaction::RuntimeState runtime_state =
        interaction::RuntimeState::Disabled;
    interaction::RuntimeDiagnostics diagnostics{};
    std::optional<interaction::InteractionTarget> target{};
    std::optional<interaction::PlacementSurface> surface{};
    uint64_t request_sequence = 0U;
    interaction::Transform displayed_root{};
    interaction::Transform simulation_root{};
    interaction::ControllerInteractionEdges pending_edges{};
};

float autodemo_planar_distance(vec3 left, vec3 right)
{
    return std::hypot(left.x - right.x, left.z - right.z);
}

bool autodemo_same_ik(
    const interaction::IKConfig& left,
    const interaction::IKConfig& right)
{
    return left.maximum_request_position_m ==
               right.maximum_request_position_m &&
        left.maximum_request_orientation_radians ==
               right.maximum_request_orientation_radians &&
        left.accepted_position_m == right.accepted_position_m &&
        left.accepted_orientation_radians ==
               right.accepted_orientation_radians &&
        left.damping == right.damping &&
        left.finite_difference_radians ==
               right.finite_difference_radians &&
        left.orientation_scale_m_per_radian ==
               right.orientation_scale_m_per_radian &&
        left.maximum_step_radians == right.maximum_step_radians &&
        left.maximum_iterations == right.maximum_iterations;
}

bool placement_pick_preview_same_vec3(vec3 left, vec3 right)
{
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool placement_pick_preview_same_quat(quat left, quat right)
{
    return left.w == right.w && left.x == right.x &&
        left.y == right.y && left.z == right.z;
}

bool placement_pick_preview_same_transform(
    const interaction::Transform& left,
    const interaction::Transform& right)
{
    return placement_pick_preview_same_vec3(
               left.position, right.position) &&
        placement_pick_preview_same_quat(
               left.rotation, right.rotation);
}

bool placement_pick_preview_same_affordance(
    const interaction::GraspAffordance& left,
    const interaction::GraspAffordance& right)
{
    return left.id == right.id && left.hand == right.hand &&
        placement_pick_preview_same_transform(
            left.hand_in_object, right.hand_in_object) &&
        placement_pick_preview_same_vec3(
            left.approach_direction_object,
            right.approach_direction_object) &&
        left.clearance_radius == right.clearance_radius;
}

bool placement_pick_preview_same_target(
    const interaction::InteractionTarget& left,
    const interaction::InteractionTarget& right)
{
    if (left.handle != right.handle ||
        !placement_pick_preview_same_transform(
            left.object_world, right.object_world) ||
        left.object_profile_id != right.object_profile_id ||
        !placement_pick_preview_same_vec3(
            left.object_bounds.center_object,
            right.object_bounds.center_object) ||
        !placement_pick_preview_same_vec3(
            left.object_bounds.half_extents_object,
            right.object_bounds.half_extents_object) ||
        !placement_pick_preview_same_vec3(
            left.object_dimensions, right.object_dimensions) ||
        !placement_pick_preview_same_transform(
            left.table_world, right.table_world) ||
        !placement_pick_preview_same_vec3(
            left.table_size, right.table_size) ||
        left.state != right.state ||
        left.owner_request != right.owner_request ||
        left.affordances.size() != right.affordances.size())
    {
        return false;
    }
    for (size_t index = 0; index < left.affordances.size(); ++index)
    {
        if (!placement_pick_preview_same_affordance(
                left.affordances[index], right.affordances[index]))
        {
            return false;
        }
    }
    return true;
}

bool placement_pick_preview_same_place_affordance(
    const interaction::PlaceAffordance& left,
    const interaction::PlaceAffordance& right)
{
    return left.id == right.id &&
        placement_pick_preview_same_transform(
            left.object_in_surface, right.object_in_surface) &&
        placement_pick_preview_same_vec3(
            left.support_point_object, right.support_point_object) &&
        placement_pick_preview_same_vec3(
            left.approach_direction_surface,
            right.approach_direction_surface) &&
        left.clearance_radius == right.clearance_radius;
}

bool placement_pick_preview_same_surface(
    const interaction::PlacementSurface& left,
    const interaction::PlacementSurface& right)
{
    if (!(left.handle == right.handle) ||
        !placement_pick_preview_same_transform(
            left.surface_world, right.surface_world) ||
        !placement_pick_preview_same_transform(
            left.support_volume_world, right.support_volume_world) ||
        !placement_pick_preview_same_vec3(
            left.support_volume_size, right.support_volume_size) ||
        left.half_extent_x_m != right.half_extent_x_m ||
        left.half_extent_z_m != right.half_extent_z_m ||
        left.overhead_clearance_m != right.overhead_clearance_m ||
        left.affordances.size() != right.affordances.size())
    {
        return false;
    }
    for (size_t index = 0; index < left.affordances.size(); ++index)
    {
        if (!placement_pick_preview_same_place_affordance(
                left.affordances[index], right.affordances[index]))
        {
            return false;
        }
    }
    return true;
}

bool placement_pick_preview_same_candidate(
    const interaction::PlaceCandidate& left,
    const interaction::PlaceCandidate& right)
{
    return left.mode == right.mode &&
        left.source_id == right.source_id &&
        left.selection_id == right.selection_id &&
        left.timing.canonical_fps == right.timing.canonical_fps &&
        left.timing.playback_speed == right.timing.playback_speed &&
        left.timing.entry_blend_seconds ==
            right.timing.entry_blend_seconds &&
        left.timing.reversed_commit_seconds ==
            right.timing.reversed_commit_seconds &&
        left.timing.maximum_alignment_seconds ==
            right.timing.maximum_alignment_seconds &&
        left.match.maximum_entry_root_error_m ==
            right.match.maximum_entry_root_error_m &&
        left.match.maximum_entry_yaw_error_radians ==
            right.match.maximum_entry_yaw_error_radians &&
        autodemo_same_ik(left.ik, right.ik) &&
        left.clip == right.clip &&
        left.entry_frame == right.entry_frame &&
        left.commit_frame == right.commit_frame &&
        left.release_frame == right.release_frame &&
        left.stop_frame == right.stop_frame &&
        left.direction == right.direction &&
        placement_pick_preview_same_transform(
            left.scene_from_source, right.scene_from_source) &&
        placement_pick_preview_same_transform(
            left.staging_root_world, right.staging_root_world) &&
        placement_pick_preview_same_vec3(
            left.entry_root_offset, right.entry_root_offset) &&
        left.entry_yaw_offset == right.entry_yaw_offset &&
        left.total_cost == right.total_cost;
}

bool placement_pick_preview_same_place_preview(
    const interaction::PlaceStagingPreview& left,
    const interaction::PlaceStagingPreview& right)
{
    return left.accepted == right.accepted &&
        left.ready == right.ready &&
        left.reason == right.reason &&
        placement_pick_preview_same_candidate(
            left.candidate, right.candidate) &&
        autodemo_same_ik(left.ik, right.ik) &&
        left.ik_config_fingerprint == right.ik_config_fingerprint &&
        placement_pick_preview_same_transform(
            left.staging_root_world, right.staging_root_world) &&
        left.root_error_m == right.root_error_m &&
        left.yaw_error_radians == right.yaw_error_radians;
}

bool placement_pick_preview_same_fit(
    const interaction::PlacementFit& left,
    const interaction::PlacementFit& right)
{
    return left.accepted == right.accepted &&
        left.reason == right.reason &&
        left.support_gap_m == right.support_gap_m &&
        left.lowest_corner_m == right.lowest_corner_m &&
        left.highest_corner_m == right.highest_corner_m &&
        left.footprint_valid == right.footprint_valid &&
        left.overhead_valid == right.overhead_valid;
}

bool placement_pick_preview_same_place_diagnostics(
    const interaction::RuntimePlaceDiagnostics& left,
    const interaction::RuntimePlaceDiagnostics& right)
{
    return left.surface == right.surface &&
        left.affordance_id == right.affordance_id &&
        left.mode == right.mode &&
        left.selection_id == right.selection_id &&
        left.preview_available == right.preview_available &&
        placement_pick_preview_same_place_preview(
            left.preview, right.preview) &&
        left.candidate_certified == right.candidate_certified &&
        left.preflight_config_identity == right.preflight_config_identity &&
        autodemo_same_ik(left.effective_ik, right.effective_ik) &&
        left.ik_config_fingerprint == right.ik_config_fingerprint &&
        placement_pick_preview_same_transform(
            left.requested_goal_world, right.requested_goal_world) &&
        placement_pick_preview_same_fit(
            left.requested_fit, right.requested_fit) &&
        placement_pick_preview_same_fit(
            left.actual_fit, right.actual_fit) &&
        left.clip == right.clip &&
        left.source_frame == right.source_frame &&
        left.source_frame_exact == right.source_frame_exact &&
        left.commit_frame == right.commit_frame &&
        left.phase == right.phase &&
        left.time_to_release_seconds == right.time_to_release_seconds &&
        left.committed == right.committed &&
        left.release_due == right.release_due &&
        left.released == right.released &&
        left.support_sweep_clear == right.support_sweep_clear &&
        left.support_position_error_m == right.support_position_error_m &&
        left.support_orientation_error_radians ==
            right.support_orientation_error_radians &&
        left.requested_root_correction_m ==
            right.requested_root_correction_m &&
        left.applied_root_correction_m ==
            right.applied_root_correction_m &&
        left.requested_yaw_correction_radians ==
            right.requested_yaw_correction_radians &&
        left.applied_yaw_correction_radians ==
            right.applied_yaw_correction_radians &&
        left.requested_hand_correction_m ==
            right.requested_hand_correction_m &&
        left.applied_hand_correction_m ==
            right.applied_hand_correction_m &&
        left.requested_hand_orientation_radians ==
            right.requested_hand_orientation_radians &&
        left.applied_hand_orientation_radians ==
            right.applied_hand_orientation_radians &&
        left.reason == right.reason;
}

bool placement_pick_preview_same_diagnostics(
    const interaction::RuntimeDiagnostics& left,
    const interaction::RuntimeDiagnostics& right)
{
    return left.state == right.state &&
        left.result == right.result &&
        left.reason == right.reason &&
        left.target == right.target &&
        left.object_state == right.object_state &&
        left.affordance_id == right.affordance_id &&
        left.clip == right.clip &&
        left.frame == right.frame &&
        left.phase == right.phase &&
        left.hand == right.hand &&
        left.total_cost == right.total_cost &&
        left.group_costs == right.group_costs &&
        left.requested_root_correction_m ==
            right.requested_root_correction_m &&
        left.applied_root_correction_m ==
            right.applied_root_correction_m &&
        left.requested_yaw_correction_radians ==
            right.requested_yaw_correction_radians &&
        left.applied_yaw_correction_radians ==
            right.applied_yaw_correction_radians &&
        left.playback_speed == right.playback_speed &&
        left.hand_position_error_m == right.hand_position_error_m &&
        left.hand_orientation_error_radians ==
            right.hand_orientation_error_radians &&
        left.hand_constraint_weight == right.hand_constraint_weight &&
        left.attached == right.attached &&
        left.recorded_carry == right.recorded_carry &&
        left.inactive_arm_targets_locomotion ==
            right.inactive_arm_targets_locomotion &&
        left.inactive_arm_tracks_locomotion ==
            right.inactive_arm_tracks_locomotion &&
        left.pack_available == right.pack_available &&
        placement_pick_preview_same_place_diagnostics(
            left.place, right.place);
}

PlacementPickPreviewObservation capture_placement_pick_preview_observation(
    const interaction::InteractionRuntime& runtime,
    const interaction::TargetRegistry& target_registry,
    const interaction::PlacementSurfaceRegistry& surface_registry,
    interaction::TargetHandle target_handle,
    interaction::SurfaceHandle surface_handle,
    uint64_t request_sequence,
    interaction::Transform displayed_root,
    interaction::Transform simulation_root,
    interaction::ControllerInteractionEdges pending_edges)
{
    PlacementPickPreviewObservation observation{};
    observation.runtime_state = runtime.state();
    observation.diagnostics = runtime.diagnostics();
    const interaction::InteractionTarget* target =
        target_registry.find_by_id(target_handle.id);
    if (target != nullptr)
    {
        observation.target = *target;
    }
    const interaction::PlacementSurface* surface =
        surface_registry.find_by_id(surface_handle.id);
    if (surface != nullptr)
    {
        observation.surface = *surface;
    }
    observation.request_sequence = request_sequence;
    observation.displayed_root = displayed_root;
    observation.simulation_root = simulation_root;
    observation.pending_edges = pending_edges;
    return observation;
}

bool operator==(
    const PlacementPickPreviewObservation& left,
    const PlacementPickPreviewObservation& right)
{
    if (left.runtime_state != right.runtime_state ||
        !placement_pick_preview_same_diagnostics(
            left.diagnostics, right.diagnostics) ||
        left.target.has_value() != right.target.has_value() ||
        left.surface.has_value() != right.surface.has_value() ||
        left.request_sequence != right.request_sequence ||
        !placement_pick_preview_same_transform(
            left.displayed_root, right.displayed_root) ||
        !placement_pick_preview_same_transform(
            left.simulation_root, right.simulation_root) ||
        left.pending_edges.interact_pressed !=
            right.pending_edges.interact_pressed ||
        left.pending_edges.cancel_pressed !=
            right.pending_edges.cancel_pressed ||
        left.pending_edges.reset_pressed !=
            right.pending_edges.reset_pressed)
    {
        return false;
    }
    if (left.target.has_value() &&
        !placement_pick_preview_same_target(*left.target, *right.target))
    {
        return false;
    }
    return !left.surface.has_value() ||
        placement_pick_preview_same_surface(*left.surface, *right.surface);
}

bool operator!=(
    const PlacementPickPreviewObservation& left,
    const PlacementPickPreviewObservation& right)
{
    return !(left == right);
}

void validate_autodemo_state_progression(
    ControllerAutodemoState& state,
    interaction::RuntimeState runtime_state)
{
    if (!state.collapsed_states.empty() &&
        state.collapsed_states.back() == runtime_state)
    {
        return;
    }
    constexpr std::array<interaction::RuntimeState, 7> expected = {
        interaction::RuntimeState::Locomotion,
        interaction::RuntimeState::Preflight,
        interaction::RuntimeState::Align,
        interaction::RuntimeState::PickupReplay,
        interaction::RuntimeState::Hold,
        interaction::RuntimeState::Carry,
        interaction::RuntimeState::Locomotion};
    const size_t next = state.collapsed_states.size();
    if (next >= expected.size() || expected[next] != runtime_state)
    {
        throw std::runtime_error("autodemo runtime state regression");
    }
    state.collapsed_states.push_back(runtime_state);
}

void validate_placement_autodemo_state_progression(
    ControllerPlacementAutodemoState& state,
    const interaction::RuntimeDiagnostics& diagnostics)
{
    const interaction::RuntimeState runtime_state = diagnostics.state;
    if (!state.collapsed_states.empty() &&
        state.collapsed_states.back() == runtime_state)
    {
        return;
    }
    constexpr std::array<interaction::RuntimeState, 11> expected = {
        interaction::RuntimeState::Locomotion,
        interaction::RuntimeState::Preflight,
        interaction::RuntimeState::Align,
        interaction::RuntimeState::PickupReplay,
        interaction::RuntimeState::Hold,
        interaction::RuntimeState::Carry,
        interaction::RuntimeState::PlacePreflight,
        interaction::RuntimeState::PlaceAlign,
        interaction::RuntimeState::PlaceReplay,
        interaction::RuntimeState::PlaceRelease,
        interaction::RuntimeState::Locomotion};
    const size_t next = state.collapsed_states.size();
    if (next >= expected.size() || expected[next] != runtime_state)
    {
        std::string message =
            "placement auto-demo runtime state regression: expected=";
        message += next < expected.size()
            ? interaction::debug_draw::state_name(expected[next])
            : "<complete>";
        message += " observed=";
        message += interaction::debug_draw::state_name(runtime_state);
        message += " result=";
        message += interaction::debug_draw::result_name(diagnostics.result);
        message += " reason=";
        message += interaction::debug_draw::reason_name(diagnostics.reason);
        message += " collapsed=";
        for (size_t index = 0; index < state.collapsed_states.size(); ++index)
        {
            if (index != 0U)
            {
                message += ',';
            }
            message += interaction::debug_draw::state_name(
                state.collapsed_states[index]);
        }
        throw std::runtime_error(message);
    }
    state.collapsed_states.push_back(runtime_state);
}

void write_autodemo_record(
    std::ostream& output,
    uint64_t render_frame,
    uint64_t runtime_tick,
    int scheduler_phase,
    const interaction::RuntimeOutput& runtime_output,
    int carry_command_frame,
    vec3 root_position,
    float root_displacement_m,
    const AutodemoEvidenceCapture& capture,
    AutodemoAction action)
{
    const auto finite_quaternion = [](quat rotation)
    {
        return autodemo_is_finite(rotation.w) &&
            autodemo_is_finite(rotation.x) &&
            autodemo_is_finite(rotation.y) &&
            autodemo_is_finite(rotation.z);
    };
    const auto finite_transform = [&](const interaction::Transform& transform)
    {
        return autodemo_is_finite(transform.position.x) &&
            autodemo_is_finite(transform.position.y) &&
            autodemo_is_finite(transform.position.z) &&
            finite_quaternion(transform.rotation);
    };
    const auto unit_quaternion = [&](quat rotation)
    {
        const float magnitude = quat_length(rotation);
        return finite_quaternion(rotation) &&
            autodemo_is_finite(magnitude) &&
            std::fabs(magnitude - 1.0F) <= 1.0e-3F;
    };
    const bool active_hand_is_valid =
        capture.active_hand_joint == 18 || capture.active_hand_joint == 22;
    const interaction::RuntimeState runtime_state =
        runtime_output.diagnostics.state;
    const bool interaction_owned_state =
        runtime_state == interaction::RuntimeState::PickupReplay ||
        runtime_state == interaction::RuntimeState::Hold ||
        runtime_state == interaction::RuntimeState::Carry;
    if (scheduler_phase != 0 ||
        !autodemo_is_finite(root_position.x) ||
        !autodemo_is_finite(root_position.y) ||
        !autodemo_is_finite(root_position.z) ||
        !autodemo_is_finite(root_displacement_m) ||
        !autodemo_hand_constraint_evidence_valid(capture) ||
        !autodemo_is_finite(
            capture.hand_constraint_result.reach_shortfall_m) ||
        capture.hand_constraint_result.reach_shortfall_m < 0.0F ||
        !unit_quaternion(capture.hand_constraint_calibration_rotation) ||
        !unit_quaternion(capture.calibrated_hand_world_rotation) ||
        !finite_transform(capture.object_world) ||
        !finite_transform(capture.hand_in_object) ||
        !finite_transform(capture.grasp_world) ||
        capture.grasp_evidence_valid != active_hand_is_valid ||
        (!capture.grasp_evidence_valid &&
         capture.active_hand_joint != -1) ||
        (interaction_owned_state && !capture.grasp_evidence_valid))
    {
        throw std::runtime_error("autodemo evidence contains invalid values");
    }

    constexpr float kQuaternionNormTolerance = 1.0e-3F;
    for (size_t joint = 0;
         joint < interaction::kFlatControllerBoneCount;
         ++joint)
    {
        const vec3 position = capture.joint_positions[joint];
        const quat rotation = capture.joint_rotations[joint];
        if (!autodemo_is_finite(position.x) ||
            !autodemo_is_finite(position.y) ||
            !autodemo_is_finite(position.z) ||
            !autodemo_is_finite(rotation.w) ||
            !autodemo_is_finite(rotation.x) ||
            !autodemo_is_finite(rotation.y) ||
            !autodemo_is_finite(rotation.z))
        {
            throw std::runtime_error(
                "autodemo frame " + std::to_string(render_frame) +
                " joint " + std::to_string(joint) + " (" +
                kAutodemoFlatJointNames[joint] +
                ") contains a non-finite rendered transform");
        }
        const float norm_error = std::fabs(quat_length(rotation) - 1.0F);
        if (!autodemo_is_finite(norm_error) ||
            norm_error > kQuaternionNormTolerance)
        {
            throw std::runtime_error(
                "autodemo frame " + std::to_string(render_frame) +
                " joint " + std::to_string(joint) + " (" +
                kAutodemoFlatJointNames[joint] +
                ") rendered quaternion norm error exceeds 0.001");
        }
    }

    output << std::fixed << std::setprecision(6)
        << "{\"render_frame\":" << render_frame
        << ",\"runtime_tick\":" << runtime_tick
        << ",\"scheduler_phase\":" << scheduler_phase
        << ",\"state\":\""
        << interaction::debug_draw::state_name(
            runtime_output.diagnostics.state)
        << "\",\"result\":\""
        << interaction::debug_draw::result_name(
            runtime_output.diagnostics.result)
        << "\",\"reason\":\""
        << interaction::debug_draw::reason_name(
            runtime_output.diagnostics.reason)
        << "\",\"object_state\":\""
        << interaction::debug_draw::object_state_name(
            runtime_output.diagnostics.object_state)
        << "\",\"attached\":"
        << (runtime_output.diagnostics.attached ? "true" : "false")
        << ",\"owns_pose\":"
        << (runtime_output.owns_pose ? "true" : "false")
        << ",\"carry_mode\":\""
        << interaction::controller_carry_mode_label(runtime_output)
        << "\",\"carry_command_frame\":" << carry_command_frame
        << ",\"root_position\":["
        << root_position.x << ',' << root_position.y << ',' << root_position.z
        << "],\"object_position\":["
        << capture.object_world.position.x << ','
        << capture.object_world.position.y << ','
        << capture.object_world.position.z
        << "],\"grasp_evidence_valid\":"
        << (capture.grasp_evidence_valid ? "true" : "false")
        << ",\"active_hand_joint\":" << capture.active_hand_joint
        << ",\"hand_constraint_weight\":"
        << capture.hand_constraint_weight
        << ",\"hand_constraint_validated\":"
        << (capture.hand_constraint_validated ? "true" : "false")
        << ",\"hand_constraint_applied\":"
        << (capture.hand_constraint_result.applied ? "true" : "false")
        << ",\"hand_constraint_reachable\":"
        << (capture.hand_constraint_result.reachable ? "true" : "false")
        << ",\"hand_constraint_used_clavicle\":"
        << (capture.hand_constraint_result.used_clavicle ? "true" : "false")
        << ",\"hand_constraint_reach_shortfall_m\":"
        << capture.hand_constraint_result.reach_shortfall_m
        << ",\"hand_constraint_calibration_rotation\":["
        << capture.hand_constraint_calibration_rotation.w << ','
        << capture.hand_constraint_calibration_rotation.x << ','
        << capture.hand_constraint_calibration_rotation.y << ','
        << capture.hand_constraint_calibration_rotation.z
        << "],\"calibrated_hand_world_rotation\":["
        << capture.calibrated_hand_world_rotation.w << ','
        << capture.calibrated_hand_world_rotation.x << ','
        << capture.calibrated_hand_world_rotation.y << ','
        << capture.calibrated_hand_world_rotation.z
        << "],\"object_world_rotation\":["
        << capture.object_world.rotation.w << ','
        << capture.object_world.rotation.x << ','
        << capture.object_world.rotation.y << ','
        << capture.object_world.rotation.z
        << "],\"hand_in_object_position\":["
        << capture.hand_in_object.position.x << ','
        << capture.hand_in_object.position.y << ','
        << capture.hand_in_object.position.z
        << "],\"hand_in_object_rotation\":["
        << capture.hand_in_object.rotation.w << ','
        << capture.hand_in_object.rotation.x << ','
        << capture.hand_in_object.rotation.y << ','
        << capture.hand_in_object.rotation.z
        << "],\"grasp_world_position\":["
        << capture.grasp_world.position.x << ','
        << capture.grasp_world.position.y << ','
        << capture.grasp_world.position.z
        << "],\"grasp_world_rotation\":["
        << capture.grasp_world.rotation.w << ','
        << capture.grasp_world.rotation.x << ','
        << capture.grasp_world.rotation.y << ','
        << capture.grasp_world.rotation.z
        << "],\"root_displacement_m\":" << root_displacement_m
        << ",\"joint_world_positions\":[";
    for (size_t joint = 0;
         joint < interaction::kFlatControllerBoneCount;
         ++joint)
    {
        const vec3 position = capture.joint_positions[joint];
        output << (joint == 0U ? "" : ",") << '['
            << position.x << ',' << position.y << ',' << position.z << ']';
    }
    output << "],\"joint_world_rotations\":[";
    for (size_t joint = 0;
         joint < interaction::kFlatControllerBoneCount;
         ++joint)
    {
        const quat rotation = capture.joint_rotations[joint];
        output << (joint == 0U ? "" : ",") << '['
            << rotation.w << ',' << rotation.x << ','
            << rotation.y << ',' << rotation.z << ']';
    }
    output << "],\"action\":\"" << autodemo_action_name(action)
        << "\"}\n";
    if (!output)
    {
        throw std::runtime_error("cannot write autodemo JSONL");
    }
}

void write_placement_autodemo_record(
    std::ostream& output,
    const ControllerPlacementAutodemoState& state,
    uint64_t runtime_tick,
    int scheduler_phase,
    const interaction::RuntimeOutput& runtime_output,
    const interaction::RuntimeConfig& runtime_config,
    const interaction::InteractionTarget& source_target,
    const interaction::PlacementSurface& destination,
    uint32_t destination_affordance_id,
    interaction::ObjectLocalBounds object_bounds,
    const AutodemoEvidenceCapture& capture,
    vec3 displayed_root,
    float pickup_distance_m,
    float walking_displacement_m,
    float displayed_root_speed_mps,
    AutodemoAction action,
    const char* demo_phase,
    uint32_t phase_counter,
    int carry_staging_tick,
    bool preview_called_this_tick,
    bool destination_support_committed)
{
    const interaction::PlaceStagingPreview preview =
        state.live_preview.has_value()
        ? *state.live_preview
        : interaction::PlaceStagingPreview{};
    const interaction::PlaceAffordance* place_affordance = nullptr;
    for (const interaction::PlaceAffordance& affordance :
         destination.affordances)
    {
        if (affordance.id == destination_affordance_id)
        {
            place_affordance = &affordance;
            break;
        }
    }
    if (place_affordance == nullptr)
    {
        throw std::runtime_error(
            "placement auto-demo authored affordance disappeared");
    }
    if (source_target.affordances.size() != 1U)
    {
        throw std::runtime_error(
            "placement auto-demo source affordance disappeared");
    }
    const interaction::GraspAffordance& grasp_affordance =
        source_target.affordances.front();
    const int interaction_hand_side =
        grasp_affordance.hand == interaction::Hand::Right ? 1 : -1;
    const interaction::Transform place_goal =
        interaction::placement_goal_world(
            destination, place_affordance->object_in_surface);
    const interaction::PlacementFit requested =
        interaction::evaluate_placement_fit(
            destination, *place_affordance, object_bounds);
    const interaction::PlacementFit actual = state.actual_fit_captured
        ? state.final_actual_fit
        : interaction::PlacementFit{};
    const interaction::PlaceCandidate& candidate = preview.candidate;
    const char* place_mode = preview.accepted
        ? interaction::debug_draw::place_mode_name(candidate.mode)
        : "none";
    const char* place_source = "none";
    if (preview.accepted)
    {
        place_source = candidate.mode ==
                interaction::PlaceMotionMode::RecordedPlace
            ? "recorded_place"
            : "pickup_clip_0";
    }
    const bool place_state =
        runtime_output.diagnostics.state ==
            interaction::RuntimeState::PlacePreflight ||
        runtime_output.diagnostics.state ==
            interaction::RuntimeState::PlaceAlign ||
        runtime_output.diagnostics.state ==
            interaction::RuntimeState::PlaceReplay ||
        runtime_output.diagnostics.state ==
            interaction::RuntimeState::PlaceRelease ||
        state.place_observed;
    const char* place_phase = place_state
        ? interaction::debug_draw::place_phase_name(
              runtime_output.diagnostics.place.phase)
        : "none";
    const float place_position_error_m = length(
        capture.object_world.position - place_goal.position);
    const float place_orientation_error_degrees =
        quat_angle_between(
            capture.object_world.rotation, place_goal.rotation) *
        180.0F / PIf;
    const float reach_waypoint_position_error_m =
        autodemo_planar_distance(
            displayed_root, state.reach_waypoint.position);
    const float reach_waypoint_yaw_error_degrees = std::abs(
        autodemo_shortest_angle(
            autodemo_yaw_radians(capture.joint_rotations[0]) -
            autodemo_yaw_radians(state.reach_waypoint.rotation))) *
        180.0F / PIf;
    const float left_stick_magnitude = length(state.left_stick_command);

    const bool pick_preview_available = state.pick_preview_epoch >= 0;
    const bool pick_entry_selection_frozen =
        state.frozen_pick_entry_index.has_value();
    if (state.pick_preview_consumed_epoch < -1 ||
        state.pick_preview_consumed_epoch > state.pick_preview_epoch ||
        state.pick_preview_epoch_count >
            kPlacementAutodemoMaximumWalkTicks ||
        state.runtime_pick_preview_calls !=
            2U * state.pick_preview_epoch_count ||
        state.pick_preview_elapsed_ticks !=
            state.pick_preview_epoch_count ||
        state.pick_preview_mutation_count != 0U ||
        (pick_preview_available !=
         (state.pick_preview_epoch_count != 0U)) ||
        (pick_preview_available &&
         (state.pick_preview_epoch != static_cast<int64_t>(
              state.pick_preview_epoch_count - 1U) ||
          state.pick_preview_snapshot_fingerprint == 0U)) ||
        (!pick_preview_available &&
         state.pick_preview_snapshot_fingerprint != 0U) ||
        (pick_entry_selection_frozen !=
         state.pick_entry_freeze_tick.has_value()) ||
        (pick_entry_selection_frozen &&
         (*state.frozen_pick_entry_index >= state.pick_entry_slots.ordered.size() ||
          !pick_preview_available ||
          state.pick_preview_consumed_epoch != state.pick_preview_epoch ||
          *state.pick_entry_freeze_tick > runtime_tick ||
          !placement_pick_preview_same_transform(
              state.interaction_waypoint,
              state.pick_entry_slots
                  .ordered[*state.frozen_pick_entry_index].waypoint))) ||
        (state.pending_pick_preview_failure !=
             PlacementPickPreviewFailure::None &&
         (pick_entry_selection_frozen || !pick_preview_available ||
          state.pick_preview_consumed_epoch != state.pick_preview_epoch)) ||
        (state.pick_preview_failure_recorded !=
         (state.pending_pick_preview_failure !=
          PlacementPickPreviewFailure::None)))
    {
        throw std::runtime_error(
            "placement pickup preview evidence state is incoherent");
    }

    for (size_t i = 0; i < state.pick_entry_slots.ordered.size(); ++i)
    {
        const interaction::PickEntrySlot& slot =
            state.pick_entry_slots.ordered[i];
        const interaction::PickEntryPreview& slot_preview =
            state.pick_entry_previews[i];
        const bool cost_available = pick_preview_available &&
            (slot_preview.match_ready ||
             slot_preview.match_reason == interaction::Reason::PoorMatch);
        if ((i == 0U &&
             slot.identity != interaction::PickEntrySlotIdentity::Plus) ||
            (i == 1U &&
             slot.identity != interaction::PickEntrySlotIdentity::Minus) ||
            !autodemo_is_finite(slot.waypoint.position.x) ||
            !autodemo_is_finite(slot.waypoint.position.y) ||
            !autodemo_is_finite(slot.waypoint.position.z) ||
            !autodemo_is_finite(slot.waypoint.rotation.w) ||
            !autodemo_is_finite(slot.waypoint.rotation.x) ||
            !autodemo_is_finite(slot.waypoint.rotation.y) ||
            !autodemo_is_finite(slot.waypoint.rotation.z) ||
            !autodemo_is_finite(slot.prospective_root.world_x) ||
            !autodemo_is_finite(slot.prospective_root.world_z) ||
            !autodemo_is_finite(
                slot.prospective_root.world_yaw_radians) ||
            !autodemo_is_finite(slot.hand_score) ||
            !autodemo_is_finite(state.pick_entry_slots.clearance_chord_m) ||
            !autodemo_is_finite(
                state.pick_entry_slots.preserved_standoff_m) ||
            (pick_preview_available &&
             (slot_preview.prospective_root.world_x !=
                  slot.prospective_root.world_x ||
              slot_preview.prospective_root.world_z !=
                  slot.prospective_root.world_z ||
              slot_preview.prospective_root.world_yaw_radians !=
                  slot.prospective_root.world_yaw_radians ||
              slot_preview.path_feasible !=
                  (slot_preview.path_reason == interaction::Reason::None) ||
              slot_preview.match_ready !=
                  (slot_preview.match_reason == interaction::Reason::None))) ||
            (cost_available &&
             (!autodemo_is_finite(slot_preview.total_cost) ||
              slot_preview.total_cost < 0.0F)) ||
            (pick_entry_selection_frozen &&
             *state.frozen_pick_entry_index == i &&
             !(slot_preview.path_feasible && slot_preview.match_ready)))
        {
            throw std::runtime_error(
                "placement pickup preview slot evidence is incoherent");
        }
    }
    const interaction::PickEntrySlot& pick_plus =
        state.pick_entry_slots.ordered[0];
    const interaction::PickEntrySlot& pick_minus =
        state.pick_entry_slots.ordered[1];
    const interaction::PickEntryPreview& pick_plus_preview =
        state.pick_entry_previews[0];
    const interaction::PickEntryPreview& pick_minus_preview =
        state.pick_entry_previews[1];
    const bool pick_plus_cost_available = pick_preview_available &&
        (pick_plus_preview.match_ready ||
         pick_plus_preview.match_reason == interaction::Reason::PoorMatch);
    const bool pick_minus_cost_available = pick_preview_available &&
        (pick_minus_preview.match_ready ||
         pick_minus_preview.match_reason == interaction::Reason::PoorMatch);
    const bool pick_plus_eligible = pick_preview_available &&
        pick_plus_preview.path_feasible && pick_plus_preview.match_ready;
    const bool pick_minus_eligible = pick_preview_available &&
        pick_minus_preview.path_feasible && pick_minus_preview.match_ready;
    const size_t frozen_pick_entry_index =
        state.frozen_pick_entry_index.value_or(
            state.pick_entry_slots.ordered.size());

    if (scheduler_phase != 0 ||
        !autodemo_is_finite(pickup_distance_m) ||
        !autodemo_is_finite(walking_displacement_m) ||
        !autodemo_is_finite(displayed_root_speed_mps) ||
        !autodemo_is_finite(left_stick_magnitude) ||
        !autodemo_is_finite(state.interaction_lateral_offset_m) ||
        !autodemo_is_finite(reach_waypoint_position_error_m) ||
        !autodemo_is_finite(reach_waypoint_yaw_error_degrees) ||
        !autodemo_is_finite(state.brake_simulation_speed_mps) ||
        state.brake_simulation_speed_mps < 0.0F ||
        state.stationary_transition_count >
            state.stationary_search_calls ||
        !autodemo_hand_constraint_evidence_valid(capture) ||
        (state.stationary_search_calls == 0U &&
         (state.stationary_transition_count != 0U ||
          state.stationary_selected_frame != -1 ||
          state.stationary_selected_range != -1)) ||
        (state.stationary_search_calls != 0U &&
         (state.stationary_selected_frame < 0 ||
          state.stationary_selected_range < 0)) ||
        (runtime_output.diagnostics.attached && runtime_output.owns_pose &&
         !capture.grasp_evidence_valid))
    {
        throw std::runtime_error(
            "placement auto-demo evidence contains invalid values");
    }

    output << std::fixed << std::setprecision(6)
        << "{\"render_frame\":" << state.render_frame
        << ",\"runtime_tick\":" << runtime_tick
        << ",\"scheduler_phase\":" << scheduler_phase
        << ",\"state\":\""
        << interaction::debug_draw::state_name(
            runtime_output.diagnostics.state)
        << "\",\"result\":\""
        << interaction::debug_draw::result_name(
            runtime_output.diagnostics.result)
        << "\",\"reason\":\""
        << placement_pick_preview_reason_name(
               runtime_output.diagnostics.reason)
        << "\",\"object_state\":\""
        << interaction::debug_draw::object_state_name(
            runtime_output.diagnostics.object_state)
        << "\",\"attached\":"
        << (runtime_output.diagnostics.attached ? "true" : "false")
        << ",\"owns_pose\":"
        << (runtime_output.owns_pose ? "true" : "false")
        << ",\"action\":\"" << autodemo_action_name(action)
        << "\",\"demo_phase\":\"" << demo_phase
        << "\",\"phase_counter\":" << phase_counter
        << ",\"pickup_distance_m\":" << pickup_distance_m
        << ",\"walking_origin_displacement_m\":"
        << walking_displacement_m
        << ",\"displayed_root_speed_mps\":"
        << displayed_root_speed_mps
        << ",\"left_stick_command\":["
        << state.left_stick_command.x << ','
        << state.left_stick_command.y << ','
        << state.left_stick_command.z
        << "],\"left_stick_magnitude\":"
        << left_stick_magnitude
        << ",\"root_position\":["
        << displayed_root.x << ',' << displayed_root.y << ','
        << displayed_root.z
        << "],\"object_position\":["
        << capture.object_world.position.x << ','
        << capture.object_world.position.y << ','
        << capture.object_world.position.z
        << "],\"object_world_rotation\":["
        << capture.object_world.rotation.w << ','
        << capture.object_world.rotation.x << ','
        << capture.object_world.rotation.y << ','
        << capture.object_world.rotation.z
        << "],\"locomotion_provider_kind\":\""
        << (state.last_provider_was_live_flat
                ? "live_flat" : "canonical")
        << "\",\"canonical_snapshot_used\":"
        << (state.canonical_snapshot_provider_calls != 0U
                ? "true" : "false")
        << ",\"root_relocation_applied\":"
        << (state.root_relocation_calls != 0U ? "true" : "false")
        << ",\"simulation_root_initialized_from_reach\":"
        << (state.simulation_root_initialization_calls != 0U
                ? "true" : "false")
        << ",\"displayed_root_initialized_from_reach\":"
        << (state.displayed_root_initialization_calls != 0U
                ? "true" : "false")
        << ",\"reach_waypoint_position\":["
        << state.reach_waypoint.position.x << ','
        << state.reach_waypoint.position.y << ','
        << state.reach_waypoint.position.z
        << "],\"reach_waypoint_rotation\":["
        << state.reach_waypoint.rotation.w << ','
        << state.reach_waypoint.rotation.x << ','
        << state.reach_waypoint.rotation.y << ','
        << state.reach_waypoint.rotation.z
        << "],\"interaction_waypoint_position\":["
        << state.interaction_waypoint.position.x << ','
        << state.interaction_waypoint.position.y << ','
        << state.interaction_waypoint.position.z
        << "],\"interaction_waypoint_rotation\":["
        << state.interaction_waypoint.rotation.w << ','
        << state.interaction_waypoint.rotation.x << ','
        << state.interaction_waypoint.rotation.y << ','
        << state.interaction_waypoint.rotation.z
        << "],\"interaction_lateral_offset_m\":"
        << state.interaction_lateral_offset_m
        << ",\"interaction_object_position\":["
        << source_target.object_world.position.x << ','
        << source_target.object_world.position.y << ','
        << source_target.object_world.position.z
        << "],\"interaction_object_dimensions\":["
        << source_target.object_dimensions.x << ','
        << source_target.object_dimensions.y << ','
        << source_target.object_dimensions.z
        << "],\"interaction_object_rotation\":["
        << source_target.object_world.rotation.w << ','
        << source_target.object_world.rotation.x << ','
        << source_target.object_world.rotation.y << ','
        << source_target.object_world.rotation.z
        << "],\"interaction_approach_direction_object\":["
        << grasp_affordance.approach_direction_object.x << ','
        << grasp_affordance.approach_direction_object.y << ','
        << grasp_affordance.approach_direction_object.z
        << "],\"interaction_clearance_radius_m\":"
        << grasp_affordance.clearance_radius
        << ",\"interaction_hand_side\":"
        << interaction_hand_side
        << ",\"reach_waypoint_position_error_m\":"
        << reach_waypoint_position_error_m
        << ",\"reach_waypoint_yaw_error_degrees\":"
        << reach_waypoint_yaw_error_degrees
        << ",\"reach_brake_latched\":"
        << (state.reach_braking ? "true" : "false")
        << ",\"stationary_candidate_count\":"
        << state.stationary_candidate_count
        << ",\"stationary_constraint_active\":"
        << (state.stationary_constraint_active ? "true" : "false")
        << ",\"stationary_search_calls\":"
        << state.stationary_search_calls
        << ",\"stationary_transition_count\":"
        << state.stationary_transition_count
        << ",\"stationary_selected_frame\":"
        << state.stationary_selected_frame
        << ",\"stationary_selected_range\":"
        << state.stationary_selected_range
        << ",\"brake_simulation_speed_mps\":"
        << state.brake_simulation_speed_mps
        << ",\"pick_resolver_maximum_m\":"
        << runtime_config.matcher.maximum_approach_m
        << ",\"pick_resolver_live_flat\":"
        << (state.pick_resolver_used_live_flat ? "true" : "false")
        << ",\"destination_handle_id\":" << destination.handle.id
        << ",\"destination_generation\":"
        << destination.handle.generation
        << ",\"destination_affordance_id\":"
        << destination_affordance_id
        << ",\"destination_generation_changed_before_place\":"
        << (state.destination_generation_changed_before_place
                ? "true" : "false")
        << ",\"placement_surface_resolver_calls\":"
        << state.placement_surface_resolver_calls
        << ",\"runtime_preview_calls\":" << state.runtime_preview_calls
        << ",\"runtime_preview_source_state\":\""
        << (preview_called_this_tick ? "Carry" : "none")
        << "\",\"free_selector_calls\":" << state.free_selector_calls
        << ",\"external_match_input_calls\":"
        << state.external_match_input_calls
        << ",\"preview_mutation_count\":"
        << state.preview_mutation_count
        << ",\"far_selection_id\":"
        << state.initial_far_selection_id
        << ",\"current_selection_id\":"
        << state.current_selection_id
        << ",\"submitted_selection_id\":"
        << (action == AutodemoAction::Place
                ? state.submitted_selection_id : 0U)
        << ",\"candidate_certified\":"
        << (preview.accepted ? "true" : "false")
        << ",\"place_mode\":\"" << place_mode
        << "\",\"place_source\":\"" << place_source
        << "\",\"place_source_id\":" << candidate.source_id
        << ",\"preview_accepted\":"
        << (preview.accepted ? "true" : "false")
        << ",\"preview_ready\":"
        << (preview.ready ? "true" : "false")
        << ",\"preview_ik_fingerprint\":"
        << preview.ik_config_fingerprint
        << ",\"ik_maximum_request_position_m\":"
        << runtime_config.ik.maximum_request_position_m
        << ",\"ik_maximum_request_orientation_degrees\":"
        << runtime_config.ik.maximum_request_orientation_radians *
            180.0F / PIf
        << ",\"ik_accepted_position_m\":"
        << runtime_config.ik.accepted_position_m
        << ",\"ik_accepted_orientation_degrees\":"
        << runtime_config.ik.accepted_orientation_radians * 180.0F / PIf
        << ",\"ik_damping\":" << runtime_config.ik.damping
        << ",\"ik_finite_difference_radians\":"
        << runtime_config.ik.finite_difference_radians
        << ",\"ik_orientation_scale_m_per_radian\":"
        << runtime_config.ik.orientation_scale_m_per_radian
        << ",\"ik_maximum_step_radians\":"
        << runtime_config.ik.maximum_step_radians
        << ",\"ik_maximum_iterations\":"
        << runtime_config.ik.maximum_iterations
        << ",\"staging_root_position\":["
        << preview.staging_root_world.position.x << ','
        << preview.staging_root_world.position.y << ','
        << preview.staging_root_world.position.z
        << "],\"staging_root_rotation\":["
        << preview.staging_root_world.rotation.w << ','
        << preview.staging_root_world.rotation.x << ','
        << preview.staging_root_world.rotation.y << ','
        << preview.staging_root_world.rotation.z
        << "],\"preview_root_error_m\":" << preview.root_error_m
        << ",\"preview_yaw_error_degrees\":"
        << preview.yaw_error_radians * 180.0F / PIf
        << ",\"carry_staging_tick\":" << carry_staging_tick
        << ",\"selected_staging_root\":"
        << (preview.accepted ? "true" : "false")
        << ",\"direct_root_write\":"
        << (state.direct_root_write_calls != 0U ? "true" : "false")
        << ",\"place_phase\":\"" << place_phase
        << "\",\"place_goal_position\":["
        << place_goal.position.x << ',' << place_goal.position.y << ','
        << place_goal.position.z
        << "],\"place_goal_rotation\":["
        << place_goal.rotation.w << ',' << place_goal.rotation.x << ','
        << place_goal.rotation.y << ',' << place_goal.rotation.z
        << "],\"place_position_error_m\":" << place_position_error_m
        << ",\"place_orientation_error_degrees\":"
        << place_orientation_error_degrees
        << ",\"requested_fit_accepted\":"
        << (requested.accepted ? "true" : "false")
        << ",\"requested_support_gap_m\":" << requested.support_gap_m
        << ",\"requested_lowest_corner_m\":"
        << requested.lowest_corner_m
        << ",\"requested_highest_corner_m\":"
        << requested.highest_corner_m
        << ",\"requested_footprint_valid\":"
        << (requested.footprint_valid ? "true" : "false")
        << ",\"requested_overhead_valid\":"
        << (requested.overhead_valid ? "true" : "false")
        << ",\"actual_fit_accepted\":"
        << (actual.accepted ? "true" : "false")
        << ",\"actual_support_gap_m\":" << actual.support_gap_m
        << ",\"actual_lowest_corner_m\":" << actual.lowest_corner_m
        << ",\"actual_highest_corner_m\":" << actual.highest_corner_m
        << ",\"actual_footprint_valid\":"
        << (actual.footprint_valid ? "true" : "false")
        << ",\"actual_bound_corners_valid\":"
        << (actual.accepted && actual.footprint_valid &&
                actual.overhead_valid ? "true" : "false")
        << ",\"actual_overhead_valid\":"
        << (actual.overhead_valid ? "true" : "false")
        << ",\"support_sweep_clear\":"
        << (state.final_support_sweep_clear ? "true" : "false")
        << ",\"attachment_transition_count\":"
        << state.attachment_transition_count
        << ",\"attached_to_free_transition_count\":"
        << state.attached_to_free_transition_count
        << ",\"destination_support_committed\":"
        << (destination_support_committed ? "true" : "false")
        << ",\"grasp_evidence_valid\":"
        << (capture.grasp_evidence_valid ? "true" : "false")
        << ",\"active_hand_joint\":" << capture.active_hand_joint
        << ",\"hand_constraint_weight\":"
        << capture.hand_constraint_weight
        << ",\"hand_constraint_validated\":"
        << (capture.hand_constraint_validated ? "true" : "false")
        << ",\"hand_constraint_applied\":"
        << (capture.hand_constraint_result.applied ? "true" : "false")
        << ",\"hand_constraint_reachable\":"
        << (capture.hand_constraint_result.reachable ? "true" : "false")
        << ",\"hand_constraint_used_clavicle\":"
        << (capture.hand_constraint_result.used_clavicle ? "true" : "false")
        << ",\"hand_constraint_reach_shortfall_m\":"
        << capture.hand_constraint_result.reach_shortfall_m
        << ",\"hand_constraint_calibration_rotation\":["
        << capture.hand_constraint_calibration_rotation.w << ','
        << capture.hand_constraint_calibration_rotation.x << ','
        << capture.hand_constraint_calibration_rotation.y << ','
        << capture.hand_constraint_calibration_rotation.z
        << "],\"calibrated_hand_world_rotation\":["
        << capture.calibrated_hand_world_rotation.w << ','
        << capture.calibrated_hand_world_rotation.x << ','
        << capture.calibrated_hand_world_rotation.y << ','
        << capture.calibrated_hand_world_rotation.z
        << "],\"hand_in_object_position\":["
        << capture.hand_in_object.position.x << ','
        << capture.hand_in_object.position.y << ','
        << capture.hand_in_object.position.z
        << "],\"hand_in_object_rotation\":["
        << capture.hand_in_object.rotation.w << ','
        << capture.hand_in_object.rotation.x << ','
        << capture.hand_in_object.rotation.y << ','
        << capture.hand_in_object.rotation.z
        << "],\"grasp_world_position\":["
        << capture.grasp_world.position.x << ','
        << capture.grasp_world.position.y << ','
        << capture.grasp_world.position.z
        << "],\"grasp_world_rotation\":["
        << capture.grasp_world.rotation.w << ','
        << capture.grasp_world.rotation.x << ','
        << capture.grasp_world.rotation.y << ','
        << capture.grasp_world.rotation.z
        << "],\"joint_world_positions\":[";
    for (size_t joint = 0;
         joint < interaction::kFlatControllerBoneCount;
         ++joint)
    {
        const vec3 position = capture.joint_positions[joint];
        output << (joint == 0U ? "" : ",") << '['
            << position.x << ',' << position.y << ',' << position.z << ']';
    }
    output << "],\"joint_world_rotations\":[";
    for (size_t joint = 0;
         joint < interaction::kFlatControllerBoneCount;
         ++joint)
    {
        const quat rotation = capture.joint_rotations[joint];
        output << (joint == 0U ? "" : ",") << '['
            << rotation.w << ',' << rotation.x << ','
            << rotation.y << ',' << rotation.z << ']';
    }
    output << ']'
        << ",\"pick_preview_epoch\":" << state.pick_preview_epoch
        << ",\"pick_preview_consumed_epoch\":"
        << state.pick_preview_consumed_epoch
        << ",\"pick_preview_snapshot_fingerprint\":"
        << state.pick_preview_snapshot_fingerprint
        << ",\"pick_preview_snapshot_source\":\""
        << (pick_preview_available ? "live_flat" : "unset")
        << "\",\"pick_preview_epoch_count\":"
        << state.pick_preview_epoch_count
        << ",\"runtime_pick_preview_calls\":"
        << state.runtime_pick_preview_calls
        << ",\"pick_preview_mutation_count\":"
        << state.pick_preview_mutation_count
        << ",\"pick_preview_elapsed_ticks\":"
        << state.pick_preview_elapsed_ticks
        << ",\"pick_preview_pending_failure\":\""
        << placement_pick_preview_failure_name(
               state.pending_pick_preview_failure)
        << "\",\"pick_preview_failure_recorded\":"
        << (state.pick_preview_failure_recorded ? "true" : "false")
        << ",\"pick_entry_selected_slot\":\""
        << (pick_entry_selection_frozen
                ? interaction::pick_entry_slot_name(
                      state.pick_entry_slots
                          .ordered[frozen_pick_entry_index].identity)
                : "none")
        << "\",\"pick_entry_selection_frozen\":"
        << (pick_entry_selection_frozen ? "true" : "false")
        << ",\"pick_entry_freeze_tick\":"
        << (state.pick_entry_freeze_tick.has_value()
                ? static_cast<int64_t>(*state.pick_entry_freeze_tick)
                : -1)
        << ",\"pick_interact_submission_count\":"
        << state.pick_interact_submission_count
        << ",\"pick_reservation_transition_count\":"
        << state.pick_reservation_transition_count
        << ",\"pick_plus_identity\":\""
        << interaction::pick_entry_slot_name(pick_plus.identity)
        << "\",\"pick_plus_evaluation_order\":0"
        << ",\"pick_plus_waypoint_position\":["
        << pick_plus.waypoint.position.x << ','
        << pick_plus.waypoint.position.y << ','
        << pick_plus.waypoint.position.z
        << "],\"pick_plus_waypoint_rotation\":["
        << pick_plus.waypoint.rotation.w << ','
        << pick_plus.waypoint.rotation.x << ','
        << pick_plus.waypoint.rotation.y << ','
        << pick_plus.waypoint.rotation.z
        << "],\"pick_plus_prospective_world_x\":"
        << pick_plus.prospective_root.world_x
        << ",\"pick_plus_prospective_world_z\":"
        << pick_plus.prospective_root.world_z
        << ",\"pick_plus_prospective_world_yaw_radians\":"
        << pick_plus.prospective_root.world_yaw_radians
        << ",\"pick_plus_clearance_chord_m\":"
        << state.pick_entry_slots.clearance_chord_m
        << ",\"pick_plus_preserved_standoff_m\":"
        << state.pick_entry_slots.preserved_standoff_m
        << ",\"pick_plus_hand_score\":" << pick_plus.hand_score
        << ",\"pick_plus_path_feasible\":"
        << (pick_preview_available && pick_plus_preview.path_feasible
                ? "true" : "false")
        << ",\"pick_plus_match_ready\":"
        << (pick_preview_available && pick_plus_preview.match_ready
                ? "true" : "false")
        << ",\"pick_plus_path_reason\":\""
        << placement_pick_preview_reason_name(
               pick_preview_available
                   ? pick_plus_preview.path_reason
                   : interaction::Reason::NoCandidate)
        << "\",\"pick_plus_match_reason\":\""
        << placement_pick_preview_reason_name(
               pick_preview_available
                   ? pick_plus_preview.match_reason
                   : interaction::Reason::NoCandidate)
        << "\",\"pick_plus_feasible_entry_frame\":"
        << (pick_preview_available
                ? pick_plus_preview.feasible_entry_frame : -1)
        << ",\"pick_plus_contact_frame\":"
        << (pick_preview_available ? pick_plus_preview.contact_frame : -1)
        << ",\"pick_plus_cost_available\":"
        << (pick_plus_cost_available ? "true" : "false")
        << ",\"pick_plus_total_cost\":"
        << (pick_plus_cost_available ? pick_plus_preview.total_cost : 0.0F)
        << ",\"pick_plus_eligible\":"
        << (pick_plus_eligible ? "true" : "false")
        << ",\"pick_plus_selected\":"
        << (pick_entry_selection_frozen && frozen_pick_entry_index == 0U
                ? "true" : "false")
        << ",\"pick_minus_identity\":\""
        << interaction::pick_entry_slot_name(pick_minus.identity)
        << "\",\"pick_minus_evaluation_order\":1"
        << ",\"pick_minus_waypoint_position\":["
        << pick_minus.waypoint.position.x << ','
        << pick_minus.waypoint.position.y << ','
        << pick_minus.waypoint.position.z
        << "],\"pick_minus_waypoint_rotation\":["
        << pick_minus.waypoint.rotation.w << ','
        << pick_minus.waypoint.rotation.x << ','
        << pick_minus.waypoint.rotation.y << ','
        << pick_minus.waypoint.rotation.z
        << "],\"pick_minus_prospective_world_x\":"
        << pick_minus.prospective_root.world_x
        << ",\"pick_minus_prospective_world_z\":"
        << pick_minus.prospective_root.world_z
        << ",\"pick_minus_prospective_world_yaw_radians\":"
        << pick_minus.prospective_root.world_yaw_radians
        << ",\"pick_minus_clearance_chord_m\":"
        << state.pick_entry_slots.clearance_chord_m
        << ",\"pick_minus_preserved_standoff_m\":"
        << state.pick_entry_slots.preserved_standoff_m
        << ",\"pick_minus_hand_score\":" << pick_minus.hand_score
        << ",\"pick_minus_path_feasible\":"
        << (pick_preview_available && pick_minus_preview.path_feasible
                ? "true" : "false")
        << ",\"pick_minus_match_ready\":"
        << (pick_preview_available && pick_minus_preview.match_ready
                ? "true" : "false")
        << ",\"pick_minus_path_reason\":\""
        << placement_pick_preview_reason_name(
               pick_preview_available
                   ? pick_minus_preview.path_reason
                   : interaction::Reason::NoCandidate)
        << "\",\"pick_minus_match_reason\":\""
        << placement_pick_preview_reason_name(
               pick_preview_available
                   ? pick_minus_preview.match_reason
                   : interaction::Reason::NoCandidate)
        << "\",\"pick_minus_feasible_entry_frame\":"
        << (pick_preview_available
                ? pick_minus_preview.feasible_entry_frame : -1)
        << ",\"pick_minus_contact_frame\":"
        << (pick_preview_available ? pick_minus_preview.contact_frame : -1)
        << ",\"pick_minus_cost_available\":"
        << (pick_minus_cost_available ? "true" : "false")
        << ",\"pick_minus_total_cost\":"
        << (pick_minus_cost_available ? pick_minus_preview.total_cost : 0.0F)
        << ",\"pick_minus_eligible\":"
        << (pick_minus_eligible ? "true" : "false")
        << ",\"pick_minus_selected\":"
        << (pick_entry_selection_frozen && frozen_pick_entry_index == 1U
                ? "true" : "false")
        << "}\n";
    if (!output)
    {
        throw std::runtime_error(
            "cannot write placement auto-demo JSONL");
    }
}

void validate_autodemo_screenshot(
    const AutodemoConfiguration& configuration)
{
    std::error_code error;
    const uintmax_t size = std::filesystem::file_size(
        configuration.screenshot_temporary, error);
    if (error || size <= 10000U)
    {
        throw std::runtime_error(
            "autodemo screenshot is missing or too small");
    }
}

}  // namespace

//--------------------------------------

void update_callback(void* args)
{
    ((std::function<void()>*)args)->operator()();
}

int main(void)
{
    std::optional<AutodemoConfiguration> autodemo_configuration;
    interaction::SmartPickupProductionConfig smart_pickup_configuration{};
    std::filesystem::path matching_features_output =
        "./resources/features.bin";
    try
    {
        smart_pickup_configuration =
            interaction::load_smart_pickup_production_config();
        autodemo_configuration = parse_autodemo_environment();
#if !defined(PLATFORM_WEB)
        if (autodemo_configuration.has_value())
        {
            autodemo_sigterm_requested = 0;
            if (std::signal(SIGTERM, autodemo_sigterm_handler) == SIG_ERR)
            {
                throw std::runtime_error(
                    "cannot install autodemo SIGTERM handler");
            }
        }
#endif
        const char* features_output_environment =
            std::getenv("MM_FEATURES_OUTPUT");
        if (features_output_environment != nullptr &&
            features_output_environment[0] != '\0')
        {
            matching_features_output =
                std::filesystem::path(features_output_environment);
        }
        autodemo_require_parent_directory(
            matching_features_output, "matching features");
        if (autodemo_configuration.has_value())
        {
            const std::filesystem::path normalized_features =
                autodemo_normalized_path(matching_features_output);
            const std::array<std::filesystem::path, 6> evidence_paths = {
                autodemo_configuration->log_final,
                autodemo_configuration->log_temporary,
                autodemo_configuration->log_backup,
                autodemo_configuration->screenshot_final,
                autodemo_configuration->screenshot_temporary,
                autodemo_configuration->screenshot_backup};
            for (const std::filesystem::path& evidence_path : evidence_paths)
            {
                if (normalized_features ==
                    autodemo_normalized_path(evidence_path))
                {
                    throw std::runtime_error(
                        "MM_FEATURES_OUTPUT must differ from evidence paths");
                }
            }
            prepare_autodemo_temporaries(*autodemo_configuration);
        }
    }
    catch (const std::exception& error)
    {
        if (autodemo_configuration.has_value())
        {
            cleanup_autodemo_temporaries(*autodemo_configuration);
        }
        std::fprintf(stderr, "controller: %s\n", error.what());
        return 1;
    }

    const bool pickup_autodemo_enabled =
        autodemo_configuration.has_value() &&
        autodemo_configuration->mode == AutodemoMode::Pickup;
    const bool placement_autodemo_enabled =
        autodemo_configuration.has_value() &&
        autodemo_configuration->mode == AutodemoMode::Placement;
    const bool legacy_interaction_fixture_mode =
        pickup_autodemo_enabled || placement_autodemo_enabled;

    // Init Window
    
    const int screen_width = 1280;
    const int screen_height = 720;
    
    SetConfigFlags(FLAG_VSYNC_HINT);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screen_width, screen_height, "raylib [data vs code driven displacement]");
    SetTargetFPS(25);
    if (autodemo_configuration.has_value() && !IsWindowReady())
    {
        cleanup_autodemo_temporaries(*autodemo_configuration);
        std::fprintf(stderr, "controller: autodemo window initialization failed\n");
        CloseWindow();
        return 1;
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
    
    // Scene Obstacles
    
    array1d<vec3> obstacles_positions(3);
    array1d<vec3> obstacles_scales(3);
    
    obstacles_positions(0) = vec3(5.0f, 0.0f, 6.0f);
    obstacles_positions(1) = vec3(-3.0f, 0.0f, -5.0f);
    obstacles_positions(2) = vec3(-8.0f, 0.0f, 3.0f);
    
    obstacles_scales(0) = vec3(2.0f, 1.0f, 5.0f);
    obstacles_scales(1) = vec3(4.0f, 1.0f, 4.0f);
    obstacles_scales(2) = vec3(2.0f, 1.0f, 2.0f);
    
    // Ground Plane
    
    Shader ground_plane_shader = LoadShader("./resources/checkerboard.vs", "./resources/checkerboard.fs");
    Mesh ground_plane_mesh = GenMeshPlane(20.0f, 20.0f, 10, 10);
    Model ground_plane_model = LoadModelFromMesh(ground_plane_mesh);
    ground_plane_model.materials[0].shader = ground_plane_shader;
    
    // Character
    
    // G1: no character.bin skinned mesh — the skeleton is drawn directly from
    // bone transforms in the render loop, so mesh/shader loading is skipped.

    // Load Animation Data and build Matching Database
    
    database db;
    database_load(db, "./resources/database.bin");
    if (db.nbones() !=
        static_cast<int>(interaction::kFlatControllerBoneCount))
    {
        if (autodemo_configuration.has_value())
        {
            cleanup_autodemo_temporaries(*autodemo_configuration);
        }
        std::fprintf(
            stderr,
            "controller: ordinary database must contain exactly %zu bones\n",
            interaction::kFlatControllerBoneCount);
        UnloadModel(ground_plane_model);
        UnloadShader(ground_plane_shader);
        CloseWindow();
        return 1;
    }
    bool flat_parent_tree_matches =
        db.bone_parents.size ==
        static_cast<int>(interaction::kFlatControllerBoneCount);
    for (size_t bone = 0;
         flat_parent_tree_matches &&
             bone < interaction::kFlatControllerBoneCount;
         ++bone)
    {
        if (db.bone_parents(static_cast<int>(bone)) !=
                interaction::kFlatControllerParents[bone])
        {
            flat_parent_tree_matches = false;
            break;
        }
    }
    if (!flat_parent_tree_matches)
    {
        if (autodemo_configuration.has_value())
        {
            cleanup_autodemo_temporaries(*autodemo_configuration);
        }
        std::fprintf(
            stderr,
            "controller: ordinary database parent tree does not match flat controller\n");
        UnloadModel(ground_plane_model);
        UnloadShader(ground_plane_shader);
        CloseWindow();
        return 1;
    }
    
    float feature_weight_foot_position = 0.75f;
    float feature_weight_foot_velocity = 1.0f;
    float feature_weight_hip_velocity = 1.0f;
    float feature_weight_trajectory_positions = 1.0f;
    float feature_weight_trajectory_directions = 1.5f;
    
    database_build_matching_features(
        db,
        feature_weight_foot_position,
        feature_weight_foot_velocity,
        feature_weight_hip_velocity,
        feature_weight_trajectory_positions,
        feature_weight_trajectory_directions);

    std::vector<int> stationary_candidates;
    try
    {
        stationary_candidates =
            stationary_motion_matching::derive_candidates(db);
        save_matching_features_checked(db, matching_features_output);
    }
    catch (const std::exception& error)
    {
        if (autodemo_configuration.has_value())
        {
            cleanup_autodemo_temporaries(*autodemo_configuration);
        }
        std::fprintf(stderr, "controller: %s\n", error.what());
        UnloadModel(ground_plane_model);
        UnloadShader(ground_plane_shader);
        CloseWindow();
        return 1;
    }

#if !defined(PLATFORM_WEB)
    try
    {
#endif
    // Interaction data is a separate fixed-25 pack. Keep these values alive
    // for the full controller lifetime because InteractionRuntime stores
    // pointers to the loaded database and features.
    std::optional<interaction::Database> interaction_database;
    std::optional<interaction::Features> interaction_features;
    interaction::TargetRegistry interaction_registry;
    interaction::PlacementSurfaceRegistry interaction_surface_registry;
    interaction::PlaceMotionLibrary interaction_place_library{};
    interaction::RuntimeConfig interaction_config{};
    interaction::TargetHandle interaction_scene_target_handle{};
    interaction::SurfaceHandle interaction_destination_surface_handle{};
    uint32_t interaction_destination_affordance_id = 0U;
    interaction::InteractionTarget interaction_authored_target{};
    interaction::PlacementSurface interaction_authored_destination_surface{};
    bool interaction_pack_loaded = false;
    std::string interaction_pack_diagnostic;

    const char* interaction_pack_environment =
        std::getenv("MM_INTERACTION_PACK");
    const bool interaction_pack_explicit =
        interaction_pack_environment != nullptr &&
        interaction_pack_environment[0] != '\0';
    if (!interaction_pack_explicit && !legacy_interaction_fixture_mode)
    {
        throw std::runtime_error(
            "manual Smart Pickup requires an explicit MM_INTERACTION_PACK");
    }
    const std::filesystem::path interaction_pack_path =
        interaction_pack_explicit
        ? std::filesystem::path(interaction_pack_environment)
        : std::filesystem::path("./resources/g1_interaction");

    interaction::InteractionRuntime interaction_runtime = [&]()
        -> interaction::InteractionRuntime {
        try
        {
            interaction_database.emplace(interaction::load_database(
                interaction_pack_path / "interaction_database.bin"));
            interaction_features.emplace(interaction::load_features(
                interaction_pack_path / "interaction_features.bin"));
            interaction::validate_controller_interaction_pack(
                *interaction_database, *interaction_features);

            // Manual Smart Pickup full-pack guard begins.
            if (!legacy_interaction_fixture_mode)
            {
                if (interaction_database->fps_numerator != 25U ||
                    interaction_database->fps_denominator != 1U ||
                    interaction_database->clip_count != 2045U ||
                    interaction_database->frame_count != 511250U ||
                    interaction_features->frame_count !=
                        interaction_database->frame_count)
                {
                    throw std::runtime_error(
                        "manual Smart Pickup requires the certified full pack");
                }
            }
            // Manual Smart Pickup full-pack guard ends.

            interaction::InteractionTarget demo_target =
                legacy_interaction_fixture_mode
                ? interaction::make_controller_demo_target(
                    *interaction_database)
                : interaction::make_smart_pickup_demo_target();
            interaction_scene_target_handle =
                interaction_registry.upsert(std::move(demo_target));
            const interaction::InteractionTarget* registered_target =
                interaction_registry.find(interaction_scene_target_handle);
            assert(registered_target != nullptr);
            interaction_authored_target = *registered_target;

            interaction::PlacementSurface destination_surface =
                legacy_interaction_fixture_mode
                ? interaction::make_controller_demo_destination_surface(
                    *interaction_database, interaction_authored_target)
                : interaction::make_smart_pickup_demo_destination_surface(
                    interaction_authored_target);
            assert(destination_surface.affordances.size() == 1U);
            interaction_destination_affordance_id =
                destination_surface.affordances.front().id;
            interaction_destination_surface_handle =
                interaction_surface_registry.upsert(destination_surface);
            const interaction::PlacementSurface* registered_destination =
                interaction_surface_registry.find(
                    interaction_destination_surface_handle);
            assert(registered_destination != nullptr);
            interaction_authored_destination_surface =
                *registered_destination;
            interaction_pack_loaded = true;
            return interaction::InteractionRuntime(
                *interaction_database,
                *interaction_features,
                interaction_registry,
                interaction_surface_registry,
                interaction_place_library,
                interaction_config);
        }
        catch (const interaction::FormatError& error)
        {
            if (!legacy_interaction_fixture_mode)
            {
                throw;
            }
            interaction_pack_diagnostic = error.what();
            interaction_pack_loaded = false;
            return interaction::InteractionRuntime::disabled(
                interaction::Reason::PackUnavailable);
        }
        catch (const std::exception& error)
        {
            if (!legacy_interaction_fixture_mode)
            {
                throw;
            }
            interaction_pack_diagnostic = error.what();
            interaction_pack_loaded = false;
            return interaction::InteractionRuntime::disabled(
                interaction::Reason::PackUnavailable);
        }
        catch (...)
        {
            if (!legacy_interaction_fixture_mode)
            {
                throw;
            }
            interaction_pack_diagnostic =
                "unknown interaction pack initialization failure";
            interaction_pack_loaded = false;
            return interaction::InteractionRuntime::disabled(
                interaction::Reason::PackUnavailable);
        }
    }();

    std::optional<AutodemoCanonicalEntry> autodemo_canonical_entry;
    ControllerAutodemoState autodemo_state;
    ControllerPlacementAutodemoState placement_autodemo_state;
    if (autodemo_configuration.has_value())
    {
        try
        {
            if (!interaction_pack_loaded || !interaction_database.has_value() ||
                !interaction_features.has_value())
            {
                throw std::runtime_error(
                    interaction_pack_diagnostic.empty()
                        ? "autodemo requires a valid interaction pack"
                        : interaction_pack_diagnostic);
            }
            if (pickup_autodemo_enabled)
            {
                autodemo_canonical_entry = make_autodemo_canonical_entry(
                    *interaction_database,
                    *interaction_features,
                    interaction_authored_target);
            }
            // Placement auto-demo setup begins.
            if (placement_autodemo_enabled)
            {
                placement_autodemo_state.stationary_candidate_count =
                    stationary_candidates.size();
                if (stationary_candidates.empty())
                {
                    throw std::runtime_error(
                        "placement auto-demo requires stationary locomotion candidates");
                }
                placement_autodemo_state.reach_waypoint =
                    interaction::make_pick_reach_waypoint(
                        *interaction_database,
                        interaction_authored_target);
                placement_autodemo_state.pick_entry_slots =
                    interaction::make_pick_entry_slots(
                        placement_autodemo_state.reach_waypoint,
                        interaction_authored_target);
                placement_autodemo_state.interaction_lateral_offset_m =
                    placement_autodemo_state.pick_entry_slots
                        .clearance_chord_m;
                vec3 reach_facing = quat_mul_vec3(
                    placement_autodemo_state.reach_waypoint.rotation,
                    vec3(0.0F, 0.0F, 1.0F));
                reach_facing.y = 0.0F;
                if (length(reach_facing) <= 1.0e-5F)
                {
                    throw std::runtime_error(
                        "placement Reach waypoint has no planar facing");
                }
                reach_facing = normalize(reach_facing);
                placement_autodemo_state.reach_entry_point =
                    placement_autodemo_state.reach_waypoint.position -
                    kPlacementAutodemoReachEntryDistanceM * reach_facing;
            }
            // Placement auto-demo setup ends.
        }
        catch (const std::exception& error)
        {
            if (autodemo_state.log.is_open())
            {
                autodemo_state.log.close();
            }
            cleanup_autodemo_temporaries(*autodemo_configuration);
            std::fprintf(stderr, "controller: %s\n", error.what());
            UnloadModel(ground_plane_model);
            UnloadShader(ground_plane_shader);
            CloseWindow();
            return 1;
        }
    }

    // The flat controller has no local channels for G1's intermediate hip,
    // shoulder, wrist, or hand links. Seed those links from a valid pack pose
    // when available; disabled interaction uses the identity-rotation neutral
    // Pose default instead. Dynamic channels are deliberately neutral here.
    interaction::Pose interaction_reference_pose{};
    if (interaction_pack_loaded && interaction_database.has_value())
    {
        const int32_t reference_frame =
            interaction_database->range_starts.at(0);
        interaction_reference_pose = interaction::pose_at_frame(
            *interaction_database, reference_frame);
        interaction_reference_pose.velocities.fill(vec3());
        interaction_reference_pose.angular_velocities.fill(vec3());
        interaction_reference_pose.hand_dof =
            interaction::kFlatControllerRestHandDof;
        interaction_reference_pose.hand_dof_velocities =
            interaction::kFlatControllerRestHandDofVelocities;
        interaction_reference_pose.foot_contacts = {};
    }
   
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
    
    bool ik_enabled = true;
    float ik_max_length_buffer = 0.015f;
    float ik_foot_height = 0.02f;
    float ik_toe_length = 0.15f;
    float ik_unlock_radius = 0.2f;
    float ik_blending_halflife = 0.1f;
    
    // Contact and Foot Locking data
    
    // The ordinary flat/LAFAN controller database uses toe indices 5 and 9.
    array1d<int> contact_bones(2);
    contact_bones(0) =
        static_cast<int>(interaction::kFlatControllerLeftToe);
    contact_bones(1) =
        static_cast<int>(interaction::kFlatControllerRightToe);
    
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

    auto reset_controller_contacts = [&]()
    {
        for (int i = 0; i < contact_bones.size; ++i)
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
    };
    
    array1d<vec3> adjusted_bone_positions = bone_positions;
    array1d<quat> adjusted_bone_rotations = bone_rotations;
    
    // Learned Motion Matching
    
    const bool lmm_enabled = false;
    
    nnet decompressor, stepper, projector;    
    nnet_load(decompressor, "./resources/decompressor.bin");
    nnet_load(stepper, "./resources/stepper.bin");
    nnet_load(projector, "./resources/projector.bin");

    nnet_evaluation decompressor_evaluation, stepper_evaluation, projector_evaluation;
    decompressor_evaluation.resize(decompressor);
    stepper_evaluation.resize(stepper);
    projector_evaluation.resize(projector);

    array1d<float> features_proj = db.features(frame_index);
    array1d<float> features_curr = db.features(frame_index);
    array1d<float> latent_proj(32); latent_proj.zero();
    array1d<float> latent_curr(32); latent_curr.zero();
    
    // Go

    const float dt = interaction::kControllerStepSeconds;
    interaction::ControllerInteractionScheduler interaction_scheduler;
    interaction::SmartPickupController manual_smart_pickup_controller(
        smart_pickup_configuration);
    interaction::SmartPickupPostStepResult manual_smart_pickup_post_step{};
    std::optional<interaction::PickRequest> manual_smart_pickup_request{};
    ManualPickStationaryDiagnostics manual_pick_stationary_diagnostics{};
    vec3 manual_pick_previous_displayed_root = bone_positions(0);
    bool manual_pick_stationary_constraint_was_active = false;
    interaction::ControllerInteractionFrameHandoff interaction_frame_handoff;
    interaction::ControllerInteractionSceneHandoff interaction_scene_handoff;
    uint64_t interaction_next_request_id = 1U;
    uint64_t smart_pickup_controller_tick = 0U;
    interaction::ControllerInteractionFrameState interaction_frame_state{};

    auto resolve_autodemo_place_target =
        [&](const interaction::LocomotionSnapshot&)
            -> std::optional<interaction::ControllerPlaceTarget>
    {
        if (!placement_autodemo_enabled || !interaction_pack_loaded ||
            interaction_destination_surface_handle.id == 0U ||
            interaction_destination_surface_handle.generation == 0U ||
            interaction_destination_affordance_id == 0U)
        {
            throw std::logic_error(
                "placement auto-demo destination identity is unavailable");
        }
        if (interaction_surface_registry.find(
                interaction_destination_surface_handle) == nullptr)
        {
            placement_autodemo_state
                .destination_generation_changed_before_place = true;
            throw std::logic_error(
                "placement auto-demo destination generation changed");
        }
        return interaction::ControllerPlaceTarget{
            interaction_destination_surface_handle,
            interaction_destination_affordance_id,
            interaction_next_request_id++};
    };

    auto resolve_manual_place_target =
        [&](const interaction::LocomotionSnapshot& snapshot)
            -> std::optional<interaction::ControllerPlaceTarget>
    {
        if (!interaction_pack_loaded)
        {
            return std::nullopt;
        }
        if (autodemo_configuration.has_value())
        {
            if (placement_autodemo_enabled)
            {
                ++placement_autodemo_state
                      .placement_surface_resolver_calls;
            }
            throw std::logic_error(
                "autodemo invoked the manual place resolver");
        }
        const std::optional<interaction::SurfaceHandle> surface_handle =
            interaction_surface_registry.resolve_single_surface(
                snapshot.pose.positions[0], 1.00F);
        if (!surface_handle.has_value())
        {
            return std::nullopt;
        }
        const interaction::PlacementSurface* surface =
            interaction_surface_registry.find(*surface_handle);
        if (surface == nullptr || surface->affordances.size() != 1U)
        {
            return std::nullopt;
        }
        return interaction::ControllerPlaceTarget{
            *surface_handle,
            surface->affordances.front().id,
            interaction_next_request_id++};
    };

    auto runtime_place_preview = [&interaction_runtime](
        interaction::SurfaceHandle surface,
        uint32_t affordance_id)
    {
        return interaction_runtime.preview_place(surface, affordance_id);
    };

    auto preview_pick_entry_slots = [&interaction_runtime](
        const interaction::LocomotionSnapshot& snapshot,
        const interaction::PickEntrySlots& slots,
        interaction::TargetHandle target,
        uint32_t affordance_id)
    {
        std::array<interaction::PickEntryPreview, 2> previews{};
        for (size_t index = 0; index < previews.size(); ++index)
        {
            previews[index] = interaction_runtime.preview_pick(
                snapshot,
                slots.ordered[index].prospective_root,
                target,
                affordance_id);
        }
        return previews;
    };

    auto make_flat_controller_pose = [&]()
    {
        interaction::FlatControllerPose pose;
        for (size_t bone = 0;
             bone < interaction::kFlatControllerBoneCount;
             ++bone)
        {
            const int index = static_cast<int>(bone);
            pose.positions[bone] = bone_positions(index);
            pose.velocities[bone] = bone_velocities(index);
            pose.rotations[bone] = bone_rotations(index);
            pose.angular_velocities[bone] =
                bone_angular_velocities(index);
        }
        pose.foot_contacts[0] = curr_bone_contacts(0) ? 1U : 0U;
        pose.foot_contacts[1] = curr_bone_contacts(1) ? 1U : 0U;
        return pose;
    };

    auto initialize_autodemo_canonical_world = [&]()
    {
        if (!autodemo_configuration.has_value())
        {
            return;
        }
        if (placement_autodemo_enabled)
        {
            ++placement_autodemo_state.root_relocation_calls;
            ++placement_autodemo_state
                  .simulation_root_initialization_calls;
            ++placement_autodemo_state
                  .displayed_root_initialization_calls;
        }
        if (!autodemo_canonical_entry.has_value())
        {
            throw std::runtime_error(
                "autodemo canonical world entry is unavailable");
        }

        const interaction::Pose& canonical_pose =
            autodemo_canonical_entry->snapshot.pose;
        constexpr size_t root = g1_skeleton::Simulation;

        inertialize_root_adjust(
            bone_offset_positions(0),
            transition_src_position,
            transition_src_rotation,
            transition_dst_position,
            transition_dst_rotation,
            bone_positions(0),
            bone_rotations(0),
            canonical_pose.positions[root],
            canonical_pose.rotations[root]);
        bone_velocities(0) = canonical_pose.velocities[root];
        bone_angular_velocities(0) =
            canonical_pose.angular_velocities[root];

        simulation_position = canonical_pose.positions[root];
        simulation_velocity = canonical_pose.velocities[root];
        simulation_acceleration = vec3();
        simulation_rotation = canonical_pose.rotations[root];
        simulation_angular_velocity =
            canonical_pose.angular_velocities[root];
        desired_velocity = simulation_velocity;
        desired_rotation = simulation_rotation;
        desired_velocity_change_curr = vec3();
        desired_velocity_change_prev = vec3();
        desired_rotation_change_curr = vec3();
        desired_rotation_change_prev = vec3();

        trajectory_positions(0) = simulation_position;
        trajectory_velocities(0) = simulation_velocity;
        trajectory_accelerations(0) = vec3();
        trajectory_rotations(0) = simulation_rotation;
        trajectory_angular_velocities(0) =
            simulation_angular_velocity;
        trajectory_desired_velocities(0) = simulation_velocity;
        trajectory_desired_rotations(0) = simulation_rotation;
        for (size_t index = 0;
             index < autodemo_canonical_entry->snapshot
                 .future_root_positions.size();
             ++index)
        {
            const int trajectory_index = static_cast<int>(index) + 1;
            trajectory_positions(trajectory_index) =
                autodemo_canonical_entry->snapshot
                    .future_root_positions[index];
            trajectory_rotations(trajectory_index) =
                autodemo_canonical_entry->snapshot
                    .future_root_rotations[index];
            trajectory_velocities(trajectory_index) =
                autodemo_canonical_entry->future_root_velocities[index];
            trajectory_angular_velocities(trajectory_index) =
                autodemo_canonical_entry
                    ->future_root_angular_velocities[index];
            trajectory_accelerations(trajectory_index) = vec3();
            trajectory_desired_velocities(trajectory_index) =
                trajectory_velocities(trajectory_index);
            trajectory_desired_rotations(trajectory_index) =
                trajectory_rotations(trajectory_index);
        }

        adjusted_bone_positions(0) = bone_positions(0);
        adjusted_bone_rotations(0) = bone_rotations(0);
        reset_controller_contacts();
    };
    if (pickup_autodemo_enabled)
    {
        initialize_autodemo_canonical_world();
    }
    const interaction::FlatControllerPose
        interaction_flat_reference_pose = [&]()
    {
        interaction::FlatControllerPose reference =
            make_flat_controller_pose();
        reference.velocities.fill(vec3());
        reference.angular_velocities.fill(vec3());
        reference.foot_contacts = {};
        return reference;
    }();
    constexpr size_t interaction_root = g1_skeleton::Simulation;
    interaction_reference_pose.velocities.fill(vec3());
    interaction_reference_pose.angular_velocities.fill(vec3());
    interaction_reference_pose.positions[interaction_root] =
        interaction_flat_reference_pose.positions[0];
    interaction_reference_pose.velocities[interaction_root] =
        interaction_flat_reference_pose.velocities[0];
    interaction_reference_pose.rotations[interaction_root] =
        interaction_flat_reference_pose.rotations[0];
    interaction_reference_pose.angular_velocities[interaction_root] =
        interaction_flat_reference_pose.angular_velocities[0];

#ifdef MM_DISCRETE
    // Optional env overrides so we can sweep halflife without recompiling.
    if (const char* e = getenv("MM_HALFLIFE"))  inertialize_blending_halflife = atof(e);
    if (const char* e = getenv("MM_SIMROT_HL")) simulation_rotation_halflife  = atof(e);
    if (const char* e = getenv("MM_STRAFE"))    g_force_strafe = (atoi(e) != 0);
    const char* logpath = getenv("MM_LOG");
    g_log = fopen(logpath ? logpath : "/home/ubuntu/projects/motion-matching/discrete_log.txt", "w");
    if (!g_log) g_log = stderr;
    fprintf(g_log, "# MM_DISCRETE run: hold-forward + azimuth snaps at f=50,100,150\n");
    fprintf(g_log, "# inertialize_blending_halflife=%.3f sim_rot_halflife=%.3f strafe=%d\n",
        inertialize_blending_halflife, simulation_rotation_halflife, (int)g_force_strafe);
#endif

    if (autodemo_configuration.has_value())
    {
        autodemo_state.log.imbue(std::locale::classic());
        autodemo_state.log.open(
            autodemo_configuration->log_temporary,
            std::ios::out | std::ios::binary | std::ios::trunc);
        if (!autodemo_state.log)
        {
            cleanup_autodemo_temporaries(*autodemo_configuration);
            std::fprintf(
                stderr, "controller: cannot open autodemo temporary JSONL\n");
            UnloadModel(ground_plane_model);
            UnloadShader(ground_plane_shader);
            CloseWindow();
            return 1;
        }
    }

    auto update_func = [&]()
    {
        AutodemoAction autodemo_action = AutodemoAction::None;
        int autodemo_carry_command_frame = -1;
        bool placement_arrival_facing_override = false;
        bool placement_brake_latched_this_tick = false;
        bool placement_pick_interact_submitted_this_tick = false;

#ifdef MM_DISCRETE
        // Camera-azimuth scripting. MM_MODE selects the pattern:
        //   0 (default): a few big 90-deg snaps (arrow taps)
        //   1: rapid alternating +/-90 snaps every N frames (arrow mashing)
        //   2: continuous azimuth ramp (arrow held), 2 rad/s like the real cam
        static int mode = -2;
        static int snapN = 5;
        if (mode == -2) { const char* m=getenv("MM_MODE"); mode=m?atoi(m):0;
                          const char* n=getenv("MM_SNAPN"); if(n) snapN=atoi(n); }
        if (mode == 0)
        {
            if (g_frame == 50) camera_azimuth += 0.5f * PIf;
            if (g_frame == 100) camera_azimuth += 0.5f * PIf;
            if (g_frame == 150) camera_azimuth -= 0.5f * PIf;
        }
        else if (mode == 1)
        {
            if (g_frame >= 25 && (g_frame % snapN) == 0)
                camera_azimuth += ((g_frame / snapN) % 2 ? -1.0f : 1.0f) * 0.5f * PIf;
        }
        else if (mode == 2)
        {
            if (g_frame >= 25) camera_azimuth += 2.0f * dt; // arrow held
        }
        else if (mode == 3)
        {
            // alternating 180-deg azimuth snaps -> antipodal desired_rotation
            if (g_frame >= 25 && (g_frame % snapN) == 0) camera_azimuth += PIf;
        }
#endif

        // Get gamepad stick states
        vec3 gamepadstick_left = gamepad_get_stick(GAMEPAD_STICK_LEFT);
        vec3 gamepadstick_right = gamepad_get_stick(GAMEPAD_STICK_RIGHT);
        const vec3 raw_gamepadstick_left = gamepadstick_left;
        const vec3 raw_gamepadstick_right = gamepadstick_right;
        const bool raw_desired_strafe = desired_strafe_update();

        // Press edges and runtime updates share the fixed 25 Hz controller
        // tick. Camera input remains live while interaction output suppresses
        // movement steering.
        interaction::ControllerInteractionEdges interaction_edges{
            IsKeyPressed(KEY_F) || IsGamepadButtonPressed(
                GAMEPAD_PLAYER, GAMEPAD_BUTTON_RIGHT_FACE_LEFT),
            IsKeyPressed(KEY_X) || IsGamepadButtonPressed(
                GAMEPAD_PLAYER, GAMEPAD_BUTTON_RIGHT_FACE_UP),
            IsKeyPressed(KEY_R) || IsGamepadButtonPressed(
                GAMEPAD_PLAYER, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT)};
        const bool manual_smart_pickup_override_pressed =
            IsKeyPressed(KEY_W) || IsKeyPressed(KEY_A) ||
            IsKeyPressed(KEY_S) || IsKeyPressed(KEY_D);
        const interaction::RuntimeOutput& cached_interaction_output =
            interaction_scheduler.cached_output();
        const interaction::RuntimeState cached_interaction_state =
            cached_interaction_output.diagnostics.state;
        const bool committed_pickup_manual_override =
            manual_smart_pickup_override_pressed &&
            !cached_interaction_output.diagnostics.attached &&
            (cached_interaction_state == interaction::RuntimeState::Align ||
             cached_interaction_state ==
                 interaction::RuntimeState::PickupReplay);
        if (committed_pickup_manual_override)
        {
            interaction_edges.cancel_pressed = true;
            gamepadstick_left = raw_gamepadstick_left;
        }
        if (cached_interaction_output.suppress_steering &&
            !committed_pickup_manual_override)
        {
            gamepadstick_left = vec3();
        }
        if (pickup_autodemo_enabled)
        {
            // Auto evidence is deterministic in the absence of external
            // input. It drives only the ordinary left-stick seam and the
            // existing scheduler edge seam.
            gamepadstick_left = vec3();
            if (autodemo_state.reset_presentation_frames_remaining > 0U)
            {
                interaction_edges = {};
            }
            if (autodemo_state.evidence_started &&
                autodemo_state.reset_presentation_frames_remaining == 0U)
            {
                if (autodemo_state.render_frame == kAutodemoInteractFrame &&
                    !autodemo_state.interact_pulsed)
                {
                    interaction_edges.interact_pressed = true;
                    autodemo_state.interact_pulsed = true;
                    autodemo_action = AutodemoAction::Interact;
                }
                else if (autodemo_state.reset_pending)
                {
                    interaction_edges.reset_pressed = true;
                    autodemo_action = AutodemoAction::Reset;
                }
                else if (autodemo_state.carry_origin_captured &&
                         autodemo_state.carry_command_count <
                             kAutodemoCarryCommandCount)
                {
                    gamepadstick_left = vec3(0.0F, 0.0F, -1.0F);
                    autodemo_carry_command_frame =
                        autodemo_state.carry_command_count;
                    autodemo_action = AutodemoAction::Forward;
                }
            }
        }
        else if (placement_autodemo_enabled)
        {
            // Placement auto-demo input begins.
            gamepadstick_left = vec3();
            gamepadstick_right = vec3();
            interaction_edges = {};
            placement_autodemo_state.left_stick_command = vec3();
            if (placement_autodemo_state.handoff_frames_remaining == 0U)
            {
                if (placement_autodemo_state.pending_pick_preview_failure !=
                        PlacementPickPreviewFailure::None &&
                    placement_autodemo_state.pick_preview_failure_recorded)
                {
                    throw std::runtime_error(
                        placement_autodemo_state
                                    .pending_pick_preview_failure ==
                                PlacementPickPreviewFailure::NoPath
                            ? "placement pickup preview found no feasible path"
                            : "placement pickup preview exceeded 250 ticks");
                }
                const interaction::RuntimeState cached_state =
                    interaction_scheduler.cached_output().diagnostics.state;
                const vec3 displayed_root = bone_positions(0);
                const float pickup_distance_m = autodemo_planar_distance(
                    displayed_root,
                    interaction_authored_target.object_world.position);
                const float walking_displacement_m =
                    placement_autodemo_state.origin_captured
                    ? autodemo_planar_distance(
                          displayed_root,
                          placement_autodemo_state.walking_origin)
                    : 0.0F;
                const bool in_standoff_band =
                    pickup_distance_m >=
                        kPlacementAutodemoArrivalConfig.minimum_standoff_m &&
                    pickup_distance_m <=
                        kPlacementAutodemoArrivalConfig.maximum_standoff_m;
                const bool walk_contract_met =
                    placement_autodemo_state.walk_ticks >=
                        kPlacementAutodemoMinimumWalkTicks &&
                    walking_displacement_m >=
                        kPlacementAutodemoMinimumWalkDisplacementM;
                const float reach_entry_error_m =
                    autodemo_planar_distance(
                        displayed_root,
                        placement_autodemo_state.reach_entry_point);
                if (!placement_autodemo_state.reach_entry_reached &&
                    reach_entry_error_m <=
                        kPlacementAutodemoReachEntryToleranceM)
                {
                    placement_autodemo_state.reach_entry_reached = true;
                }

                // Placement pickup preview consume begins.
                if (!placement_autodemo_state
                         .frozen_pick_entry_index.has_value() &&
                    placement_autodemo_state.pending_pick_preview_failure ==
                        PlacementPickPreviewFailure::None &&
                    placement_autodemo_state.pick_preview_epoch >
                        placement_autodemo_state
                            .pick_preview_consumed_epoch)
                {
                    placement_autodemo_state.pick_preview_consumed_epoch =
                        placement_autodemo_state.pick_preview_epoch;
                    const bool any_path_feasible =
                        placement_autodemo_state
                                .pick_entry_previews[0].path_feasible ||
                        placement_autodemo_state
                                .pick_entry_previews[1].path_feasible;
                    if (!any_path_feasible)
                    {
                        placement_autodemo_state
                            .pending_pick_preview_failure =
                                PlacementPickPreviewFailure::NoPath;
                    }
                    else
                    {
                        const std::optional<size_t> selected =
                            interaction::choose_pick_entry_slot(
                                placement_autodemo_state.pick_entry_slots,
                                placement_autodemo_state.pick_entry_previews,
                                {true, true},
                                interaction_authored_target
                                    .affordances.front().hand);
                        if (selected.has_value())
                        {
                            placement_autodemo_state
                                .frozen_pick_entry_index = selected;
                            placement_autodemo_state.interaction_waypoint =
                                placement_autodemo_state.pick_entry_slots
                                    .ordered[*selected].waypoint;
                            placement_autodemo_state.pick_entry_freeze_tick =
                                autodemo_state.runtime_tick + 1U;
                        }
                        else if (placement_autodemo_state
                                     .pick_preview_elapsed_ticks >=
                                 kPlacementAutodemoMaximumWalkTicks)
                        {
                            placement_autodemo_state
                                .pending_pick_preview_failure =
                                    PlacementPickPreviewFailure::Deadline;
                        }
                    }
                }

                const float interaction_position_error_m =
                    autodemo_planar_distance(
                        displayed_root,
                        placement_autodemo_state
                            .interaction_waypoint.position);
                const float interaction_yaw_error_radians = std::abs(
                    autodemo_shortest_angle(
                        autodemo_yaw_radians(bone_rotations(0)) -
                        autodemo_yaw_radians(
                            placement_autodemo_state
                                .interaction_waypoint.rotation)));
                const float planar_simulation_speed_mps = std::hypot(
                    simulation_velocity.x, simulation_velocity.z);

                // Placement arrival latch begins.
                if (placement_autodemo_state
                        .frozen_pick_entry_index.has_value() &&
                    placement_autodemo_state.pending_pick_preview_failure ==
                        PlacementPickPreviewFailure::None &&
                    !placement_autodemo_state.reach_braking &&
                    !placement_autodemo_state.interact_pulsed &&
                    cached_state == interaction::RuntimeState::Locomotion &&
                    interaction::arrival_ready(
                        interaction_position_error_m,
                        planar_simulation_speed_mps,
                        interaction_yaw_error_radians,
                        pickup_distance_m))
                {
                    placement_autodemo_state.reach_braking = true;
                    placement_autodemo_state.brake_simulation_speed_mps =
                        planar_simulation_speed_mps;
                    placement_brake_latched_this_tick = true;
                    placement_arrival_facing_override = false;
                    gamepadstick_left = vec3();
                    gamepadstick_right = vec3();
                }
                // Placement arrival latch ends.
                const bool reach_alignment_ready =
                    placement_autodemo_state
                        .frozen_pick_entry_index.has_value() &&
                    placement_autodemo_state.reach_entry_reached &&
                    interaction_position_error_m <=
                        kPlacementAutodemoAlignmentPositionErrorM &&
                    interaction_yaw_error_radians <=
                        kPlacementAutodemoReachYawErrorRadians;

                if (placement_autodemo_state.pending_pick_preview_failure ==
                        PlacementPickPreviewFailure::None &&
                    !placement_autodemo_state.interact_pulsed &&
                    cached_state == interaction::RuntimeState::Locomotion)
                {
                    if (placement_autodemo_state
                            .frozen_pick_entry_index.has_value() &&
                        placement_autodemo_state.settle_ticks >=
                            kPlacementAutodemoSettleTicks &&
                        walk_contract_met && in_standoff_band &&
                        reach_alignment_ready &&
                        placement_autodemo_state
                                .pick_interact_submission_count == 0U)
                    {
                        placement_pick_interact_submitted_this_tick = true;
                        interaction_edges.interact_pressed = true;
                        placement_autodemo_state.interact_pulsed = true;
                        ++placement_autodemo_state.pick_interact_submission_count;
                        autodemo_action = AutodemoAction::Interact;
                    }
                    else if (!placement_autodemo_state.reach_braking)
                    {
                        if (placement_autodemo_state
                                .frozen_pick_entry_index.has_value())
                        {
                            // Placement frozen arrival steering begins.
                            gamepadstick_left =
                                interaction::arrival_navigation_stick(
                                    placement_autodemo_state
                                        .interaction_waypoint.position,
                                    displayed_root,
                                    camera_azimuth);
                            vec3 interaction_forward = quat_mul_vec3(
                                placement_autodemo_state
                                    .interaction_waypoint.rotation,
                                vec3(0.0F, 0.0F, 1.0F));
                            interaction_forward.y = 0.0F;
                            gamepadstick_right =
                                interaction::arrival_facing_stick(
                                    interaction_forward,
                                    camera_azimuth);
                            placement_arrival_facing_override = true;
                            // Placement frozen arrival steering ends.
                            autodemo_action = AutodemoAction::Approach;
                        }
                        else if (!placement_autodemo_state
                                      .reach_entry_reached)
                        {
                            // Placement unfrozen entry steering begins.
                            gamepadstick_left =
                                controller_world_navigation_stick(
                                    placement_autodemo_state
                                        .reach_entry_point,
                                    displayed_root,
                                    camera_azimuth,
                                    kPlacementAutodemoNavigationSlowRadiusM);
                            if (length(gamepadstick_left) <= 1.0e-4F)
                            {
                                vec3 reach_facing = quat_mul_vec3(
                                    placement_autodemo_state
                                        .reach_waypoint.rotation,
                                    vec3(0.0F, 0.0F, 1.0F));
                                reach_facing.y = 0.0F;
                                gamepadstick_left =
                                    controller_world_navigation_stick(
                                        displayed_root +
                                            0.05F * normalize(reach_facing),
                                        displayed_root,
                                        camera_azimuth,
                                        kPlacementAutodemoNavigationSlowRadiusM);
                            }
                            // Placement unfrozen entry steering ends.
                            autodemo_action = AutodemoAction::Approach;
                        }
                    }

                    // Placement post-entry diagnostic begins.
                    if (placement_autodemo_state
                            .frozen_pick_entry_index.has_value() &&
                        !placement_autodemo_state.interact_pulsed)
                    {
                        ++placement_autodemo_state.post_entry_ticks;
                        if (placement_autodemo_state.post_entry_ticks == 250U)
                        {
                            const float displayed_speed_mps =
                                placement_autodemo_state.origin_captured
                                ? autodemo_planar_distance(
                                      displayed_root,
                                      placement_autodemo_state
                                          .previous_displayed_root) /
                                      dt
                                : 0.0F;
                            std::string message =
                                "placement auto-demo post-entry exceeded 250 ticks:";
                            message += " interaction_error_m=" +
                                std::to_string(interaction_position_error_m);
                            message += " standoff_m=" +
                                std::to_string(pickup_distance_m);
                            message += " displayed_speed_mps=" +
                                std::to_string(displayed_speed_mps);
                            message += " simulation_speed_mps=" +
                                std::to_string(planar_simulation_speed_mps);
                            message += " yaw_error_degrees=" +
                                std::to_string(
                                    interaction_yaw_error_radians *
                                    180.0F / PIf);
                            message += " brake_latched=";
                            message += placement_autodemo_state.reach_braking
                                ? "true" : "false";
                            message += " stationary_search_calls=" +
                                std::to_string(
                                    placement_autodemo_state
                                        .stationary_search_calls);
                            message += " stationary_transition_count=" +
                                std::to_string(
                                    placement_autodemo_state
                                        .stationary_transition_count);
                            message += " stationary_selected_frame=" +
                                std::to_string(
                                    placement_autodemo_state
                                        .stationary_selected_frame);
                            message += " stationary_selected_range=" +
                                std::to_string(
                                    placement_autodemo_state
                                        .stationary_selected_range);
                            throw std::runtime_error(message);
                        }
                    }
                    // Placement post-entry diagnostic ends.
                }
                // Placement pickup preview consume ends.
                else if (cached_state ==
                             interaction::RuntimeState::Carry &&
                         !placement_autodemo_state.destination_latched)
                {
                    interaction_edges.interact_pressed = true;
                    placement_autodemo_state.destination_latched = true;
                    autodemo_action = AutodemoAction::LatchDestination;
                }
                else if (cached_state ==
                             interaction::RuntimeState::Carry &&
                         interaction_scheduler.place_preview().has_value())
                {
                    gamepadstick_left = controller_place_staging_stick(
                        *interaction_scheduler.place_preview(),
                        interaction::Transform{
                            displayed_root, bone_rotations(0)},
                        camera_azimuth);
                    autodemo_action = AutodemoAction::Stage;
                }
            }
            placement_autodemo_state.left_stick_command =
                gamepadstick_left;
            // Placement auto-demo input ends.
        }
        // Manual pick-assist input begins.
        interaction::SmartPickupPreStepResult
            manual_smart_pickup_pre_step{};
        if (!legacy_interaction_fixture_mode)
        {
            const interaction::InteractionTarget* manual_smart_pickup_target =
                interaction_registry.find(
                    interaction_scene_target_handle);
            interaction::SmartPickupPreStepInput
                manual_smart_pickup_pre_input{};
            manual_smart_pickup_pre_input.runtime_state =
                cached_interaction_state;
            manual_smart_pickup_pre_input.interact_pressed =
                interaction_edges.interact_pressed;
            manual_smart_pickup_pre_input.cancel_pressed =
                interaction_edges.cancel_pressed ||
                interaction_edges.reset_pressed;
            manual_smart_pickup_pre_input.manual_override_pressed =
                manual_smart_pickup_override_pressed;
            manual_smart_pickup_pre_input.selected_target =
                manual_smart_pickup_target;
            if (manual_smart_pickup_target != nullptr &&
                manual_smart_pickup_target->affordances.size() == 1U)
            {
                manual_smart_pickup_pre_input.selected_affordance_id =
                    manual_smart_pickup_target->affordances.front().id;
            }
            manual_smart_pickup_pre_input.left_stick = gamepadstick_left;
            manual_smart_pickup_pre_input.right_stick = gamepadstick_right;
            manual_smart_pickup_pre_input.force_strafe =
                raw_desired_strafe;
            const interaction::PickAssistState
                manual_smart_pickup_state_before_pre_step =
                    manual_smart_pickup_controller.diagnostics().state;
            manual_smart_pickup_pre_step =
                manual_smart_pickup_controller.pre_step(
                    manual_smart_pickup_pre_input);
            if (manual_smart_pickup_pre_step.cancel_consumed)
            {
                interaction_edges.cancel_pressed = false;
            }
            if (manual_smart_pickup_pre_step.interact_consumed)
            {
                interaction_edges.interact_pressed = false;
            }
            const bool manual_smart_pickup_new_attempt =
                manual_smart_pickup_pre_step.interact_consumed &&
                (manual_smart_pickup_state_before_pre_step ==
                     interaction::PickAssistState::Idle ||
                 manual_smart_pickup_state_before_pre_step ==
                     interaction::PickAssistState::Submitted ||
                 manual_smart_pickup_state_before_pre_step ==
                     interaction::PickAssistState::Failed);
            if (manual_smart_pickup_pre_step.cancel_consumed ||
                manual_smart_pickup_pre_step.manual_override_consumed ||
                manual_smart_pickup_new_attempt)
            {
                manual_pick_stationary_diagnostics = {};
            }
            gamepadstick_left = manual_smart_pickup_pre_step.left_stick;
            gamepadstick_right = manual_smart_pickup_pre_step.right_stick;
        }
        // Manual pick-assist input ends.
        if (!autodemo_configuration.has_value() &&
            !interaction_edges.cancel_pressed &&
            !interaction_edges.reset_pressed &&
            interaction_scheduler.cached_output().diagnostics.state ==
                interaction::RuntimeState::Carry &&
            interaction_scheduler.place_preview().has_value())
        {
            gamepadstick_left = controller_place_staging_stick(
                *interaction_scheduler.place_preview(),
                interaction::Transform{
                    simulation_position, simulation_rotation},
                camera_azimuth);
        }

        // Get if strafe is desired
        bool desired_strafe = raw_desired_strafe;
#ifdef MM_DISCRETE
        desired_strafe = g_force_strafe;
#endif
        if (placement_arrival_facing_override)
        {
            desired_strafe = true;
        }
        // Manual pick-assist prior output begins.
        if (manual_smart_pickup_pre_step.force_strafe)
        {
            desired_strafe = true;
        }
        // Manual pick-assist prior output ends.
        
        // Get the desired gait (walk / run)
        desired_gait_update(
            desired_gait,
            desired_gait_velocity,
            dt);
        
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
          desired_strafe);
        
        trajectory_rotations_predict(
            trajectory_rotations,
            trajectory_angular_velocities,
            simulation_rotation,
            simulation_angular_velocity,
            trajectory_desired_rotations,
            simulation_rotation_halflife);
        
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
          simulation_back_speed);
        
        trajectory_positions_predict(
            trajectory_positions,
            trajectory_velocities,
            trajectory_accelerations,
            simulation_position,
            simulation_velocity,
            simulation_acceleration,
            trajectory_desired_velocities,
            simulation_velocity_halflife,
            obstacles_positions,
            obstacles_scales);
           
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
        
        assert(offset == db.nfeatures());

        // Check if we reached the end of the current anim
        bool end_of_anim = database_trajectory_index_clamp(db, frame_index, 1) == frame_index;

        // Placement stationary search begins.
        placement_autodemo_state.stationary_constraint_active =
            placement_autodemo_enabled &&
            placement_autodemo_state.reach_braking &&
            interaction_scheduler.cached_output().diagnostics.state ==
                interaction::RuntimeState::Locomotion &&
            (!placement_autodemo_state.interact_pulsed ||
             placement_pick_interact_submitted_this_tick);
        const bool manual_pick_stationary_constraint_active =
            !legacy_interaction_fixture_mode &&
            !manual_smart_pickup_pre_step.cancel_consumed &&
            !manual_smart_pickup_pre_step.manual_override_consumed &&
            !interaction_edges.reset_pressed &&
            manual_smart_pickup_post_step.assist_output.stationary_constraint &&
            cached_interaction_state ==
                interaction::RuntimeState::Locomotion;
        const bool manual_pick_stationary_constraint_latched_this_tick =
            manual_pick_stationary_constraint_active &&
            !manual_pick_stationary_constraint_was_active;
        manual_pick_stationary_constraint_was_active =
            manual_pick_stationary_constraint_active;
        const bool interaction_stationary_constraint_active =
            placement_autodemo_state.stationary_constraint_active ||
            manual_pick_stationary_constraint_active;
        const bool motion_search_due =
            force_search || search_timer <= 0.0F || end_of_anim ||
            placement_brake_latched_this_tick ||
            manual_pick_stationary_constraint_latched_this_tick;

        // Do we need to search?
#ifdef MM_DISCRETE
        int   dbg_best_index = frame_index;   // -1 == no search this frame
        bool  dbg_did_search = motion_search_due;
        bool  dbg_did_transition = false;
        quat  dbg_root_before = bone_rotations(0);
        quat  dbg_off_before  = bone_offset_rotations(0);
        quat  dbg_trns_dst_rot = trns_bone_rotations(0);
#endif
        if (motion_search_due)
        {
            if (lmm_enabled &&
                !interaction_stationary_constraint_active)
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

                if (interaction_stationary_constraint_active)
                {
                    const stationary_motion_matching::SearchResult result =
                        stationary_motion_matching::search(
                            db, query, stationary_candidates);
                    best_index = result.frame;
                    best_cost = result.cost;
                    // Manual stationary diagnostics search begins.
                    if (manual_pick_stationary_constraint_active)
                    {
                        ++manual_pick_stationary_diagnostics.search_count;
                        manual_pick_stationary_diagnostics.selected_frame =
                            best_index;
                        manual_pick_stationary_diagnostics.selected_cost =
                            best_cost;
                    }
                    // Manual stationary diagnostics search ends.
                    // Placement stationary evidence counters begin.
                    if (placement_autodemo_state
                            .stationary_constraint_active)
                    {
                        ++placement_autodemo_state.stationary_search_calls;
                        placement_autodemo_state.stationary_selected_frame =
                            best_index;
                        placement_autodemo_state.stationary_selected_range =
                            -1;
                        for (int range = 0; range < db.nranges(); ++range)
                        {
                            if (best_index >= db.range_starts(range) &&
                                best_index < db.range_stops(range))
                            {
                                placement_autodemo_state
                                    .stationary_selected_range = range;
                                break;
                            }
                        }
                        if (placement_autodemo_state
                                .stationary_selected_range < 0)
                        {
                            throw std::logic_error(
                                "stationary locomotion selection escaped clip ranges");
                        }
                    }
                    // Placement stationary evidence counters end.
                }
                else
                {
                    database_search(
                        best_index,
                        best_cost,
                        db,
                        query);
                }
                
                // Transition if better frame found, or guarantee that the
                // first constrained selection traverses the inertializer once.
                
                if (best_index != frame_index ||
                    (placement_autodemo_state.stationary_constraint_active &&
                     placement_autodemo_state
                         .stationary_transition_count == 0U) ||
                    manual_pick_stationary_constraint_latched_this_tick)
                {
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

                    if (placement_autodemo_state
                            .stationary_constraint_active)
                    {
                        ++placement_autodemo_state
                              .stationary_transition_count;
                    }
                    // Manual stationary diagnostics transition begins.
                    if (manual_pick_stationary_constraint_active)
                    {
                        ++manual_pick_stationary_diagnostics.transition_count;
                    }
                    // Manual stationary diagnostics transition ends.
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
        // Placement stationary search ends.

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
            frame_index++; // Assumes dt is fixed to 25 Hz.
            
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
        
        // Update Simulation
        
        vec3 simulation_position_prev = simulation_position;
        
        simulation_positions_update(
            simulation_position, 
            simulation_velocity, 
            simulation_acceleration,
            desired_velocity,
            simulation_velocity_halflife,
            dt,
            obstacles_positions,
            obstacles_scales);
            
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
          
            synchronized_position = simulation_collide_obstacles(
                simulation_position_prev,
                synchronized_position,
                obstacles_positions,
                obstacles_scales);
            
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

        const bool use_autodemo_canonical_snapshot =
            pickup_autodemo_enabled &&
            autodemo_canonical_entry.has_value() &&
            autodemo_state.interact_pulsed &&
            (cached_interaction_state ==
                 interaction::RuntimeState::Locomotion ||
             cached_interaction_state ==
                 interaction::RuntimeState::Preflight);
        // Advance locomotion and interaction synchronously once per 25 Hz
        // controller tick.
        const uint64_t placement_preview_calls_before_tick =
            placement_autodemo_state.runtime_preview_calls;
        const uint64_t placement_provider_calls_before_tick =
            placement_autodemo_state.locomotion_provider_calls;
        const interaction::FlatControllerPose flat_locomotion_pose =
            make_flat_controller_pose();
        interaction::Pose locomotion_pose =
            interaction::expand_flat_controller_pose(
                flat_locomotion_pose,
                interaction_reference_pose,
                interaction_flat_reference_pose);
        interaction::LocomotionSnapshot live_flat_snapshot{};
        live_flat_snapshot.pose = locomotion_pose;
        for (size_t index = 0;
             index < live_flat_snapshot.future_root_positions.size();
             ++index)
        {
            const int trajectory_index = static_cast<int>(index) + 1;
            live_flat_snapshot.future_root_positions[index] =
                trajectory_positions(trajectory_index);
            live_flat_snapshot.future_root_rotations[index] =
                trajectory_rotations(trajectory_index);
        }
        // Manual pick-assist observation begins.
        const uint64_t live_flat_snapshot_fingerprint =
            interaction::runtime_detail::
                locomotion_snapshot_fingerprint(live_flat_snapshot);
        const float manual_pick_displayed_planar_speed_mps =
            autodemo_planar_distance(
                bone_positions(0), manual_pick_previous_displayed_root) / dt;
        manual_pick_previous_displayed_root = bone_positions(0);
        if (!legacy_interaction_fixture_mode)
        {
            interaction::SmartPickupPostStepInput manual_smart_pickup_post_input{};
            manual_smart_pickup_post_input.controller_tick =
                smart_pickup_controller_tick;
            manual_smart_pickup_post_input.runtime_state =
                cached_interaction_state;
            manual_smart_pickup_post_input.live_flat_snapshot =
            live_flat_snapshot;
            const interaction::InteractionTarget* manual_smart_pickup_current_target =
                interaction_registry.find(
                    interaction_scene_target_handle);
            manual_smart_pickup_post_input.current_target =
                manual_smart_pickup_current_target;
            for (int obstacle_index = 0;
                 obstacle_index < obstacles_positions.size;
                 ++obstacle_index)
            {
                manual_smart_pickup_post_input.obstacle_centers.push_back(
                    obstacles_positions(obstacle_index));
                manual_smart_pickup_post_input.obstacle_sizes.push_back(
                    obstacles_scales(obstacle_index));
            }
            manual_smart_pickup_post_input.simulation_velocity =
                simulation_velocity;
            manual_smart_pickup_post_input.displayed_planar_speed_mps =
                manual_pick_displayed_planar_speed_mps;
            manual_smart_pickup_post_input.camera_azimuth = camera_azimuth;
            manual_smart_pickup_post_input.next_request_id =
                interaction_next_request_id;
            const interaction::SmartPickupPreviewCallback
                preview_manual_smart_pickup =
                    [&](const interaction::LocomotionSnapshot& snapshot,
                        interaction::PickEntryRoot prospective_root,
                        interaction::TargetHandle target,
                        uint32_t affordance_id)
                        -> std::optional<interaction::PickEntryPreview>
            {
                return interaction_runtime.preview_pick(
                    snapshot,
                    prospective_root,
                    target,
                    affordance_id);
            };
            manual_smart_pickup_post_step =
                manual_smart_pickup_controller.post_step(
                    manual_smart_pickup_post_input,
                    preview_manual_smart_pickup);
            assert(
                manual_smart_pickup_post_step.snapshot_fingerprint ==
                live_flat_snapshot_fingerprint);
            if (manual_smart_pickup_post_step.snapshot_fingerprint !=
                live_flat_snapshot_fingerprint)
            {
                throw std::logic_error(
                    "manual Smart Pickup snapshot fingerprint mismatch");
            }
            if (manual_smart_pickup_post_step.pick_request.has_value())
            {
                bool certified = true;
                if (manual_smart_pickup_post_step.certified_preview.has_value())
                {
                    certified = interaction_runtime.cache_certified_pick(
                        *manual_smart_pickup_post_step.pick_request,
                        *manual_smart_pickup_post_step.certified_preview);
                }
                if (certified)
                {
                    interaction_edges.interact_pressed = true;
                    manual_smart_pickup_request =
                        manual_smart_pickup_post_step.pick_request;
                }
            }
        }
        // Manual pick-assist observation ends.
        const interaction::RuntimeOutput& interaction_output =
            interaction_scheduler.tick(
                interaction_edges,
                [&]()
                {
                    if (placement_autodemo_enabled)
                    {
                        ++placement_autodemo_state
                              .locomotion_provider_calls;
                        if (use_autodemo_canonical_snapshot)
                        {
                            ++placement_autodemo_state
                                  .canonical_snapshot_provider_calls;
                            placement_autodemo_state
                                .last_provider_was_live_flat = false;
                        }
                        else
                        {
                            ++placement_autodemo_state
                                  .live_flat_provider_calls;
                            placement_autodemo_state
                                .last_provider_was_live_flat = true;
                        }
                    }
                    if (use_autodemo_canonical_snapshot)
                    {
                        return autodemo_canonical_entry->snapshot;
                    }
                    // Placement pickup preview provider begins.
                    const interaction::LocomotionSnapshot& snapshot =
                        live_flat_snapshot;
                    if (placement_autodemo_enabled &&
                        placement_autodemo_state.reach_entry_reached &&
                        !placement_autodemo_state
                             .frozen_pick_entry_index.has_value() &&
                        placement_autodemo_state
                                .pending_pick_preview_failure ==
                            PlacementPickPreviewFailure::None &&
                        !placement_autodemo_state.interact_pulsed &&
                        interaction_runtime.state() ==
                            interaction::RuntimeState::Locomotion)
                    {
                        const uint64_t fingerprint =
                            live_flat_snapshot_fingerprint;
                        const PlacementPickPreviewObservation runtime_before =
                            capture_placement_pick_preview_observation(
                                interaction_runtime,
                                interaction_registry,
                                interaction_surface_registry,
                                interaction_scene_target_handle,
                                interaction_destination_surface_handle,
                                interaction_next_request_id,
                                interaction::Transform{
                                    bone_positions(0), bone_rotations(0)},
                                interaction::Transform{
                                    simulation_position,
                                    simulation_rotation},
                                interaction_edges);

                        std::array<interaction::PickEntryPreview, 2> previews{};
                        try
                        {
                            previews = preview_pick_entry_slots(
                                snapshot,
                                placement_autodemo_state.pick_entry_slots,
                                interaction_scene_target_handle,
                                interaction_authored_target
                                    .affordances.front().id);
                        }
                        catch (...)
                        {
                            if (capture_placement_pick_preview_observation(
                                    interaction_runtime,
                                    interaction_registry,
                                    interaction_surface_registry,
                                    interaction_scene_target_handle,
                                    interaction_destination_surface_handle,
                                    interaction_next_request_id,
                                    interaction::Transform{
                                        bone_positions(0), bone_rotations(0)},
                                    interaction::Transform{
                                        simulation_position,
                                        simulation_rotation},
                                    interaction_edges) != runtime_before)
                            {
                                ++placement_autodemo_state
                                      .pick_preview_mutation_count;
                                throw std::runtime_error(
                                    "pickup preview mutated protected state");
                            }
                            throw;
                        }

                        if (capture_placement_pick_preview_observation(
                                interaction_runtime,
                                interaction_registry,
                                interaction_surface_registry,
                                interaction_scene_target_handle,
                                interaction_destination_surface_handle,
                                interaction_next_request_id,
                                interaction::Transform{
                                    bone_positions(0), bone_rotations(0)},
                                interaction::Transform{
                                    simulation_position,
                                    simulation_rotation},
                                interaction_edges) != runtime_before)
                        {
                            ++placement_autodemo_state
                                  .pick_preview_mutation_count;
                            throw std::runtime_error(
                                "pickup preview mutated protected state");
                        }

                        placement_autodemo_state.pick_entry_previews = previews;
                        placement_autodemo_state
                            .pick_preview_snapshot_fingerprint = fingerprint;
                        placement_autodemo_state.pick_preview_epoch =
                            static_cast<int64_t>(placement_autodemo_state
                                .pick_preview_epoch_count);
                        ++placement_autodemo_state.pick_preview_epoch_count;
                        placement_autodemo_state
                            .runtime_pick_preview_calls += 2U;
                        ++placement_autodemo_state.pick_preview_elapsed_ticks;
                    }
                    return snapshot;
                    // Placement pickup preview provider ends.
                },
                [&](const interaction::LocomotionSnapshot& snapshot)
                    -> std::optional<interaction::PickRequest>
                {
                    // Manual pick-assist submission begins.
                    if (manual_smart_pickup_request.has_value())
                    {
                        const std::optional<interaction::PickRequest>
                            manual_pick_request = std::exchange(
                                manual_smart_pickup_request,
                                std::nullopt);
                        if (manual_pick_request->request_id !=
                            interaction_next_request_id)
                        {
                            throw std::logic_error(
                                "manual Smart Pickup request ID mismatch");
                        }
                        ++interaction_next_request_id;
                        return manual_pick_request;
                    }
                    // Manual pick-assist submission ends.
                    if (!interaction_pack_loaded)
                    {
                        return std::nullopt;
                    }
                    if (placement_autodemo_enabled)
                    {
                        ++placement_autodemo_state.pick_resolver_calls;
                        placement_autodemo_state
                            .pick_resolver_used_live_flat =
                            placement_autodemo_state
                                .last_provider_was_live_flat &&
                            placement_autodemo_state
                                .canonical_snapshot_provider_calls == 0U;
                    }
                    if (!autodemo_configuration.has_value())
                    {
                        return std::nullopt;
                    }
                    const std::optional<interaction::TargetHandle> target_handle =
                        interaction_registry.resolve_single_target(
                            snapshot.pose.positions[0],
                            interaction_config.matcher.maximum_approach_m);
                    if (!target_handle.has_value())
                    {
                        return std::nullopt;
                    }
                    const interaction::InteractionTarget* target =
                        interaction_registry.find(*target_handle);
                    if (target == nullptr || target->affordances.size() != 1U)
                    {
                        return std::nullopt;
                    }
                    return interaction::PickRequest{
                        *target_handle,
                        target->affordances.front().id,
                        interaction_next_request_id++};
                },
                [&](const interaction::LocomotionSnapshot& snapshot)
                    -> std::optional<interaction::ControllerPlaceTarget>
                {
                    if (placement_autodemo_enabled)
                    {
                        return resolve_autodemo_place_target(snapshot);
                    }
                    return resolve_manual_place_target(snapshot);
                },
                [&](interaction::SurfaceHandle surface,
                    uint32_t affordance_id)
                {
                    if (placement_autodemo_enabled &&
                        interaction_runtime.state() !=
                            interaction::RuntimeState::Carry)
                    {
                        throw std::logic_error(
                            "placement auto-demo preview escaped Carry");
                    }
                    const interaction::RuntimeDiagnostics before =
                        interaction_runtime.diagnostics();
                    const interaction::InteractionTarget* before_target_ptr =
                        interaction_registry.find_by_id(
                            interaction_authored_target.handle.id);
                    const std::optional<interaction::InteractionTarget>
                        before_target = before_target_ptr == nullptr
                        ? std::nullopt
                        : std::optional<interaction::InteractionTarget>(
                              *before_target_ptr);
                    if (placement_autodemo_enabled)
                    {
                        ++placement_autodemo_state.runtime_preview_calls;
                    }
                    const interaction::PlaceStagingPreview preview =
                        runtime_place_preview(surface, affordance_id);
                    const interaction::RuntimeDiagnostics& after =
                        interaction_runtime.diagnostics();
                    const interaction::InteractionTarget* after_target =
                        interaction_registry.find_by_id(
                            interaction_authored_target.handle.id);
                    const bool target_changed =
                        before_target.has_value() !=
                            (after_target != nullptr) ||
                        (before_target.has_value() &&
                         (after_target->handle != before_target->handle ||
                          after_target->state != before_target->state ||
                          after_target->owner_request !=
                              before_target->owner_request ||
                          length(
                              after_target->object_world.position -
                              before_target->object_world.position) != 0.0F ||
                          quat_angle_between(
                              after_target->object_world.rotation,
                              before_target->object_world.rotation) != 0.0F));
                    const bool diagnostics_changed =
                        after.state != before.state ||
                        after.result != before.result ||
                        after.reason != before.reason ||
                        after.target != before.target ||
                        after.object_state != before.object_state ||
                        after.affordance_id != before.affordance_id ||
                        after.clip != before.clip ||
                        after.frame != before.frame ||
                        after.phase != before.phase ||
                        after.hand != before.hand ||
                        after.attached != before.attached ||
                        after.place.selection_id !=
                            before.place.selection_id ||
                        after.place.preview_available !=
                            before.place.preview_available ||
                        after.place.phase != before.place.phase ||
                        after.place.released != before.place.released;
                    if (placement_autodemo_enabled &&
                        (target_changed || diagnostics_changed))
                    {
                        ++placement_autodemo_state.preview_mutation_count;
                        throw std::logic_error(
                            "placement auto-demo preview mutated runtime or "
                            "target diagnostics");
                    }
                    return preview;
                },
                [&](const interaction::RuntimeInput& input)
                {
                    return interaction_runtime.update(input);
                });
        manual_smart_pickup_request.reset();
        if (interaction_edges.reset_pressed)
        {
            manual_smart_pickup_post_step = {};
        }
        const uint64_t placement_preview_call_delta =
            placement_autodemo_state.runtime_preview_calls -
            placement_preview_calls_before_tick;
        const uint64_t placement_provider_call_delta =
            placement_autodemo_state.locomotion_provider_calls -
            placement_provider_calls_before_tick;
        if (placement_autodemo_enabled &&
            (placement_preview_call_delta > 1U ||
             placement_provider_call_delta != 1U))
        {
            throw std::logic_error(
                "placement auto-demo callback count violated exact 25 Hz");
        }
        const bool placement_preview_called_this_tick =
            placement_autodemo_enabled &&
            placement_preview_call_delta == 1U;
        if (autodemo_configuration.has_value() &&
            interaction_scheduler.updated_last_tick())
        {
            ++autodemo_state.runtime_tick;
        }
        if (placement_autodemo_enabled &&
            interaction_scheduler.updated_last_tick())
        {
            if (interaction_scheduler.place_preview().has_value())
            {
                placement_autodemo_state.live_preview =
                    *interaction_scheduler.place_preview();
            }
            else if (interaction_output.diagnostics.state ==
                         interaction::RuntimeState::PlacePreflight &&
                     interaction_output.diagnostics.place.preview_available)
            {
                placement_autodemo_state.live_preview =
                    interaction_output.diagnostics.place.preview;
                placement_autodemo_state.submitted_selection_id =
                    interaction_output.diagnostics.place.selection_id;
                autodemo_action = AutodemoAction::Place;
            }
        }
        const bool interaction_scene_sample_updated =
            interaction_scheduler.updated_last_tick();

        // Prefer any exact selected target, then refresh its generation by
        // stable ID so resets cannot strand scene drawing.
        if (interaction_registry.find(interaction_output.diagnostics.target) !=
            nullptr)
        {
            interaction_scene_target_handle =
                interaction_output.diagnostics.target;
        }
        if (interaction_scene_target_handle.id != 0U)
        {
            const interaction::InteractionTarget* current_target =
                interaction_registry.find_by_id(
                    interaction_scene_target_handle.id);
            if (current_target != nullptr)
            {
                interaction_scene_target_handle = current_target->handle;
            }
        }

        const interaction::InteractionTarget* interaction_scene_target =
            interaction_registry.find(interaction_scene_target_handle);
        if (interaction_scene_target == nullptr && interaction_pack_loaded)
        {
            interaction_scene_target = &interaction_authored_target;
        }
        const interaction::PlacementSurface* interaction_destination_surface =
            interaction_surface_registry.find(
                interaction_destination_surface_handle);
        if (interaction_destination_surface == nullptr &&
            interaction_pack_loaded)
        {
            interaction_destination_surface =
                &interaction_authored_destination_surface;
        }
        const interaction::ControllerInteractionSceneState
            interaction_scene_state = interaction_scene_handoff.apply(
                interaction_scene_target,
                interaction_output,
                interaction_authored_target.object_world,
                0.0F,
                interaction_scene_sample_updated);

        std::optional<interaction::ControllerInteractionHandConstraint>
            interaction_hand_constraint;
        const interaction::InteractionTarget* selected_constraint_target =
            interaction_registry.find(interaction_output.diagnostics.target);
        const interaction::GraspAffordance* selected_affordance =
            interaction_registry.find_affordance(
                interaction_output.diagnostics.target,
                interaction_output.diagnostics.affordance_id);
        if (selected_constraint_target != nullptr &&
            interaction_scene_target == selected_constraint_target &&
            selected_affordance != nullptr &&
            (selected_constraint_target->state ==
                 interaction::ObjectState::Targeted ||
             selected_constraint_target->state ==
                 interaction::ObjectState::Attached ||
             selected_constraint_target->state ==
                 interaction::ObjectState::Held) &&
            selected_affordance->hand == interaction_output.diagnostics.hand)
        {
            interaction::ControllerInteractionHandConstraint constraint;
            constraint.target = interaction_output.diagnostics.target;
            constraint.affordance_id =
                interaction_output.diagnostics.affordance_id;
            constraint.hand = interaction_output.diagnostics.hand;
            constraint.grasp_world = interaction::compose(
                interaction_scene_state.object_world,
                selected_affordance->hand_in_object);
            interaction_hand_constraint = constraint;
        }

        interaction_frame_state = interaction_frame_handoff.apply(
            flat_locomotion_pose,
            interaction_output,
            interaction::kControllerStepSeconds,
            interaction_hand_constraint);
        for (size_t bone = 0;
             bone < interaction::kFlatControllerBoneCount;
             ++bone)
        {
            const int index = static_cast<int>(bone);
            bone_positions(index) =
                interaction_frame_state.pose.positions[bone];
            bone_velocities(index) =
                interaction_frame_state.pose.velocities[bone];
            bone_rotations(index) =
                interaction_frame_state.pose.rotations[bone];
            bone_angular_velocities(index) =
                interaction_frame_state.pose.angular_velocities[bone];
        }
        curr_bone_contacts(0) =
            interaction_frame_state.pose.foot_contacts[0] != 0U;
        curr_bone_contacts(1) =
            interaction_frame_state.pose.foot_contacts[1] != 0U;
        if (interaction_frame_state.synchronize_simulation_root)
        {
            simulation_position =
                interaction_frame_state.simulation_root_position;
            simulation_rotation =
                interaction_frame_state.simulation_root_rotation;
        }
        const interaction::Pose interaction_debug_pose =
            interaction::expand_flat_controller_pose(
                interaction_frame_state.pose,
                interaction_reference_pose,
                interaction_flat_reference_pose);
        
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
            if (g_frame >= 167) { fflush(g_log); fclose(g_log); _Exit(0); }
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

        std::optional<AutodemoEvidenceCapture> autodemo_evidence_capture;
        if (autodemo_configuration.has_value())
        {
            autodemo_evidence_capture = capture_autodemo_evidence(
                global_bone_positions,
                global_bone_rotations,
                interaction_output,
                interaction_frame_state,
                interaction_scene_target,
                interaction_scene_state.object_world);
        }
        
        // Update camera
        
        orbit_camera_update(
            camera, 
            camera_azimuth,
            camera_altitude,
            camera_distance,
            bone_positions(0) + vec3(0, 1, 0),
            // simulation_position + vec3(0, 1, 0),
            raw_gamepadstick_right,
            raw_desired_strafe,
            dt);

        // Render
        
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        BeginMode3D(camera);
        
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
        
        draw_obstacles(
            obstacles_positions,
            obstacles_scales);

        const std::array<vec3, 3> interaction_predicted_roots = {
            trajectory_positions(1),
            trajectory_positions(2),
            trajectory_positions(3)};
        interaction::debug_draw::draw_interaction_scene(
            interaction_scene_target,
            interaction_scene_state.object_world,
            interaction_destination_surface,
            interaction_destination_affordance_id,
            interaction_scheduler.place_preview(),
            interaction_output,
            interaction_predicted_roots,
            interaction_debug_pose,
            locomotion_pose,
            interaction_config.matcher.maximum_approach_m);
        // Manual pick-assist route rendering begins.
        const interaction::PickAssistDiagnostics& manual_pick_diagnostics =
            manual_smart_pickup_controller.diagnostics();
        if (manual_pick_diagnostics.frozen_slot_index.has_value())
        {
            const size_t frozen_slot_index =
                *manual_pick_diagnostics.frozen_slot_index;
            if (frozen_slot_index <
                manual_pick_diagnostics.slot_selection.ordered.size())
            {
                const vec3 frozen_slot =
                    manual_pick_diagnostics.slot_selection.ordered[
                        frozen_slot_index].root_world.position;
                const bool final_preview_certified =
                    (manual_pick_diagnostics.state ==
                         interaction::PickAssistState::ReadyToSubmit ||
                     manual_pick_diagnostics.state ==
                         interaction::PickAssistState::Submitted) &&
                    manual_pick_diagnostics.final_preview.available &&
                    manual_pick_diagnostics.final_preview
                        .all_preview_roots_finite &&
                    manual_pick_diagnostics.final_preview
                        .fingerprint_equal &&
                    manual_pick_diagnostics.final_preview
                        .prospective_root_equal &&
                    manual_pick_diagnostics.final_preview.path_feasible &&
                    manual_pick_diagnostics.final_preview.match_ready;
                const Color frozen_slot_color = final_preview_certified
                    ? GREEN
                    : ORANGE;
                DrawLine3D(
                    to_Vector3(bone_positions(0)),
                    to_Vector3(frozen_slot),
                    frozen_slot_color);
                DrawSphereWires(
                    to_Vector3(frozen_slot),
                    0.06F,
                    8,
                    12,
                    frozen_slot_color);
            }
        }
        // Manual pick-assist route rendering ends.

        // Learned pickup object-anchored route rendering begins.
        const std::optional<interaction::LearnedPickupDebugSnapshot>
            learned_pickup_debug =
                manual_smart_pickup_controller.learned_debug_snapshot();
        if (learned_pickup_debug.has_value() &&
            learned_pickup_debug->active)
        {
            const float route_y = bone_positions(0).y + 0.04F;
            const auto route_point = [route_y](
                const interaction::FunnelSample& sample)
            {
                return vec3(sample.x, route_y, sample.z);
            };
            const Color route_color{70, 150, 255, 255};
            const Color entry_color{0, 255, 255, 255};
            for (size_t index = 1U;
                 index < learned_pickup_debug->world_route.size();
                 ++index)
            {
                DrawLine3D(
                    to_Vector3(route_point(
                        learned_pickup_debug->world_route[index - 1U])),
                    to_Vector3(route_point(
                        learned_pickup_debug->world_route[index])),
                    route_color);
            }
            const int progress_index = std::clamp(
                learned_pickup_debug->progress_index,
                0,
                interaction::kFunnelExecutionTickCount - 1);
            const int lookahead_index = std::clamp(
                learned_pickup_debug->lookahead_index,
                0,
                interaction::kFunnelExecutionTickCount - 1);
            DrawSphereWires(
                to_Vector3(route_point(
                    learned_pickup_debug->frozen_entry_world)),
                0.055F, 8, 12, entry_color);
            DrawSphereWires(
                to_Vector3(route_point(
                    learned_pickup_debug->world_route[
                        static_cast<size_t>(progress_index)])),
                0.06F, 8, 12, GREEN);
            DrawSphereWires(
                to_Vector3(route_point(
                    learned_pickup_debug->world_route[
                        static_cast<size_t>(lookahead_index)])),
                0.065F, 8, 12, ORANGE);
            DrawSphereWires(
                to_Vector3(route_point(
                    learned_pickup_debug->terminal_root_world)),
                0.07F, 8, 12, MAGENTA);
            DrawSphereWires(
                to_Vector3(route_point(
                    learned_pickup_debug->tracked_root_world)),
                0.05F, 8, 12, WHITE);
        }
        // Learned pickup object-anchored route rendering ends.
        
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
        
        // Draw Ground Plane
        
        DrawModel(ground_plane_model, (Vector3){0.0f, -0.01f, 0.0f}, 1.0f, WHITE);
        DrawGrid(20, 1.0f);
        draw_axis(vec3(), quat());
        
        EndMode3D();

        interaction::debug_draw::draw_interaction_text(
            interaction_output,
            interaction_scheduler.place_preview(),
            interaction_pack_diagnostic.c_str(),
            340,
            20);
        // Manual pick-assist diagnostics begins.
        const interaction::PickAssistDiagnostics& diagnostics =
            manual_smart_pickup_controller.diagnostics();
        if (manual_pick_diagnostics.state ==
                interaction::PickAssistState::SlotSelectionPreview ||
            manual_pick_diagnostics.state ==
                interaction::PickAssistState::SlotApproach ||
            manual_pick_diagnostics.state ==
                interaction::PickAssistState::Settling ||
            manual_pick_diagnostics.state ==
                interaction::PickAssistState::FinalPreview ||
            manual_pick_diagnostics.state ==
                interaction::PickAssistState::ReadyToSubmit)
        {
            DrawText(
                "SMART PICKUP AUTO - WASD or X cancels",
                340,
                242,
                18,
                ORANGE);
        }
        else if (!interaction_output.diagnostics.attached &&
                 (interaction_output.diagnostics.state ==
                      interaction::RuntimeState::Align ||
                  interaction_output.diagnostics.state ==
                      interaction::RuntimeState::PickupReplay))
        {
            DrawText(
                "SMART PICKUP REACH - WASD or X cancels",
                340,
                242,
                18,
                ORANGE);
        }
        else if (interaction_output.diagnostics.attached &&
                 (interaction_output.diagnostics.state ==
                      interaction::RuntimeState::PickupReplay ||
                  interaction_output.diagnostics.state ==
                      interaction::RuntimeState::Hold))
        {
            DrawText(
                "SMART PICKUP ATTACHED - finishing recorded lift",
                340,
                242,
                18,
                GOLD);
        }
        else if (interaction_output.diagnostics.state ==
                 interaction::RuntimeState::Carry)
        {
            DrawText(
                "CARRY - WASD moves, F places, R resets",
                340,
                242,
                18,
                DARKGREEN);
        }
        else if (manual_pick_diagnostics.state ==
                 interaction::PickAssistState::Failed)
        {
            DrawText(
                TextFormat(
                    "PICKUP FAILED: %s - reposition and press F",
                    interaction::pick_assist_reason_name(
                        manual_pick_diagnostics.reason)),
                340,
                242,
                18,
                RED);
        }
        else if (interaction_output.diagnostics.state ==
                     interaction::RuntimeState::Locomotion &&
                 (interaction_output.diagnostics.result ==
                      interaction::ResultCode::Rejected ||
                  interaction_output.diagnostics.result ==
                      interaction::ResultCode::Failed))
        {
            DrawText(
                TextFormat(
                    "PICKUP FAILED: %s - reposition and press F",
                    interaction::debug_draw::reason_name(
                        interaction_output.diagnostics.reason)),
                340,
                242,
                18,
                RED);
        }
        const int manual_pick_selected_slot =
            manual_pick_diagnostics.frozen_slot_index.has_value()
            ? static_cast<int>(
                  manual_pick_diagnostics.selected_slot_id)
            : -1;
        DrawText(
            TextFormat(
                "assist=%s reason=%s slot=%d settle=%u/%u route=%.3fm "
                "origin=%.3fm bounds=%.3fm",
                interaction::pick_assist_state_name(
                    manual_pick_diagnostics.state),
                interaction::pick_assist_reason_name(
                    manual_pick_diagnostics.reason),
                manual_pick_selected_slot,
                manual_pick_diagnostics.settle_ticks,
                interaction::PickAssistConfig{}.required_settle_ticks,
                manual_pick_diagnostics.route_length_m,
                manual_pick_diagnostics.object_origin_distance_m,
                manual_pick_diagnostics.object_bounds_center_distance_m),
            340,
            262,
            14,
            DARKGRAY);
        DrawText(
            TextFormat(
                "error=%.3fm yaw=%.2fdeg speed=%.3fm/s",
                diagnostics.root_error_m,
                diagnostics.yaw_error_radians * 180.0F / PIf,
                diagnostics.speed_mps),
            340,
            282,
            14,
            DARKGRAY);
        DrawText(
            TextFormat(
                "final=%d fp_equal=%d roots_finite=%d root_equal=%d",
                manual_pick_diagnostics.final_preview.available,
                manual_pick_diagnostics.final_preview.fingerprint_equal,
                manual_pick_diagnostics.final_preview
                    .all_preview_roots_finite,
                manual_pick_diagnostics.final_preview
                    .prospective_root_equal),
            340,
            302,
            14,
            DARKGRAY);
        DrawText(
            TextFormat(
                "path=%d/%s match=%d/%s",
                manual_pick_diagnostics.final_preview.path_feasible,
                interaction::debug_draw::reason_name(
                    manual_pick_diagnostics.final_preview.path_reason),
                manual_pick_diagnostics.final_preview.match_ready,
                interaction::debug_draw::reason_name(
                    manual_pick_diagnostics.final_preview.match_reason)),
            340,
            322,
            14,
            DARKGRAY);
        DrawText(
            TextFormat(
                "entry=%d contact=%d cost=%.3f",
                manual_pick_diagnostics.final_preview.feasible_entry_frame,
                manual_pick_diagnostics.final_preview.contact_frame,
                manual_pick_diagnostics.final_preview.total_cost),
            340,
            342,
            14,
            DARKGRAY);
        DrawText(
            TextFormat(
                "stationary searches=%llu transitions=%llu frame=%d cost=%.3f",
                static_cast<unsigned long long>(
                    manual_pick_stationary_diagnostics.search_count),
                static_cast<unsigned long long>(
                    manual_pick_stationary_diagnostics.transition_count),
                manual_pick_stationary_diagnostics.selected_frame,
                manual_pick_stationary_diagnostics.selected_cost),
            340,
            362,
            14,
            DARKGRAY);
        // Manual pick-assist diagnostics ends.

        // UI
        
        //---------
        
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
            (Rectangle){ 1000, ui_lmm_hei + 10, 220, 20 },
            "disabled (25 Hz retime pending)");
        
        //---------
        
        float ui_ctrl_hei = 380;
        
        GuiGroupBox((Rectangle){ 1010, ui_ctrl_hei, 250, 140 }, "controls");
        
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  10, 200, 20 }, "Left Trigger - Strafe");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  30, 200, 20 }, "Left Stick - Move");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  50, 200, 20 }, "Right Stick - Camera / Facing (Stafe)");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  70, 200, 20 }, "Left Shoulder - Zoom In");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  90, 200, 20 }, "Right Shoulder - Zoom Out");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei + 110, 200, 20 }, "A Button - Walk");
        

        
        //---------
        
        GuiGroupBox((Rectangle){ 20, 20, 290, 190 }, "feature weights");
        
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
            
        if (GuiButton((Rectangle){ 150, 180, 120, 20 }, "rebuild database"))
        {
            database_build_matching_features(
                db,
                feature_weight_foot_position,
                feature_weight_foot_velocity,
                feature_weight_hip_velocity,
                feature_weight_trajectory_positions,
                feature_weight_trajectory_directions);
        }
        
        //---------
        
        float ui_sync_hei = 220;
        
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
        
        float ui_adj_hei = 300;
        
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
        
        float ui_clamp_hei = 440;
        
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
        
        float ui_ik_hei = 550;
        
        GuiGroupBox((Rectangle){ 20, ui_ik_hei, 290, 100 }, "inverse kinematics");
        
        bool ik_enabled_prev = ik_enabled;
        
        GuiCheckBox(
            (Rectangle){ 50, ui_ik_hei + 10, 20, 20 }, 
            "enabled",
            &ik_enabled);      
        
        // Foot locking needs resetting when IK is toggled
        if (ik_enabled && !ik_enabled_prev)
        {
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
        }
        
        GuiSliderBar(
            (Rectangle){ 150, ui_ik_hei + 40, 120, 20 }, 
            "blending halflife", 
            TextFormat("%5.3f", ik_blending_halflife), 
            &ik_blending_halflife, 0.0f, 1.0f);
        
        GuiSliderBar(
            (Rectangle){ 150, ui_ik_hei + 70, 120, 20 }, 
            "unlock radius", 
            TextFormat("%5.3f", ik_unlock_radius), 
            &ik_unlock_radius, 0.0f, 0.5f);
        
        //---------

        EndDrawing();

        if (pickup_autodemo_enabled)
        {
            if (autodemo_state.reset_presentation_frames_remaining > 0U)
            {
                --autodemo_state.reset_presentation_frames_remaining;
                if (autodemo_state.reset_presentation_frames_remaining == 0U)
                {
                    publish_autodemo_evidence(*autodemo_configuration);
                    autodemo_state.complete = true;
                    autodemo_state.exit_requested = true;
                }
                return;
            }

            ++autodemo_state.warmup_render_ticks;
            if (!autodemo_state.evidence_started)
            {
                if (interaction_output.diagnostics.state ==
                    interaction::RuntimeState::Locomotion)
                {
                    if (autodemo_state.runtime_tick == 0U)
                    {
                        throw std::runtime_error(
                            "autodemo did not warm on a 25 Hz tick");
                    }
                    autodemo_state.evidence_started = true;
                }
                else if (interaction_output.diagnostics.state !=
                         interaction::RuntimeState::Disabled)
                {
                    throw std::runtime_error(
                        "autodemo warmup produced an unexpected state");
                }
                else if (autodemo_state.warmup_render_ticks >=
                         kAutodemoWarmupFrames)
                {
                    throw std::runtime_error(
                        "autodemo warmup exceeded its 25 Hz deadline");
                }
            }

            if (autodemo_state.evidence_started)
            {
                const interaction::RuntimeState runtime_state =
                    interaction_output.diagnostics.state;
                validate_autodemo_state_progression(
                    autodemo_state, runtime_state);

                if (autodemo_state.interact_pulsed)
                {
                    const interaction::ResultCode result =
                        interaction_output.diagnostics.result;
                    if (runtime_state == interaction::RuntimeState::Disabled ||
                        result == interaction::ResultCode::Rejected ||
                        result == interaction::ResultCode::Cancelled ||
                        result == interaction::ResultCode::Failed)
                    {
                        throw std::runtime_error(
                            "autodemo interaction failed after Interact");
                    }
                }
                const bool candidate_state =
                    runtime_state == interaction::RuntimeState::Align ||
                    runtime_state ==
                        interaction::RuntimeState::PickupReplay ||
                    runtime_state == interaction::RuntimeState::Hold ||
                    runtime_state == interaction::RuntimeState::Carry;
                if (candidate_state &&
                    interaction_output.diagnostics.clip != 0)
                {
                    throw std::runtime_error(
                        "autodemo accepted a non-clip-0 candidate");
                }
                if (runtime_state == interaction::RuntimeState::Align)
                {
                    autodemo_state.candidate_verified = true;
                }

                const vec3 displayed_root = bone_positions(0);
                if (runtime_state == interaction::RuntimeState::Carry &&
                    !autodemo_state.carry_origin_captured)
                {
                    autodemo_state.carry_origin = displayed_root;
                    autodemo_state.carry_origin_captured = true;
                }
                const float root_displacement_m =
                    autodemo_state.carry_origin_captured
                    ? autodemo_planar_distance(
                          displayed_root, autodemo_state.carry_origin)
                    : 0.0F;
                if (runtime_state == interaction::RuntimeState::Carry)
                {
                    if (!interaction_output.diagnostics.attached)
                    {
                        throw std::runtime_error(
                            "autodemo Carry lost attachment");
                    }
                    const std::string carry_mode =
                        interaction::controller_carry_mode_label(
                            interaction_output);
                    if (carry_mode != "recorded" &&
                        carry_mode != "layered")
                    {
                        throw std::runtime_error(
                            "autodemo Carry has no valid carry mode");
                    }
                    autodemo_state.last_carry_displacement_m =
                        root_displacement_m;
                }

                if (autodemo_action == AutodemoAction::Forward)
                {
                    if (runtime_state != interaction::RuntimeState::Carry ||
                        autodemo_carry_command_frame !=
                            autodemo_state.carry_command_count ||
                        autodemo_carry_command_frame < 0 ||
                        autodemo_carry_command_frame >=
                            kAutodemoCarryCommandCount)
                    {
                        throw std::runtime_error(
                            "autodemo forward command escaped Carry");
                    }
                    ++autodemo_state.carry_command_count;
                    if (autodemo_state.carry_command_count ==
                        kAutodemoCarryCommandCount)
                    {
                        if (autodemo_carry_command_frame !=
                            kAutodemoFinalCarryCommand)
                        {
                            throw std::runtime_error(
                                "autodemo final Carry command is invalid");
                        }
                        const std::string screenshot_temporary =
                            autodemo_configuration->screenshot_temporary
                                .string();
                        TakeScreenshot(screenshot_temporary.c_str());
                        validate_autodemo_screenshot(
                            *autodemo_configuration);
                        autodemo_state.screenshot_captured = true;
                        autodemo_state.reset_pending = true;
                    }
                }
                else if (autodemo_carry_command_frame != -1)
                {
                    throw std::runtime_error(
                        "autodemo command frame has no forward action");
                }

                if (autodemo_action == AutodemoAction::Reset)
                {
                    if (runtime_state !=
                            interaction::RuntimeState::Locomotion ||
                        interaction_output.diagnostics.result !=
                            interaction::ResultCode::Reset ||
                        interaction_output.diagnostics.reason !=
                            interaction::Reason::Reset)
                    {
                        throw std::runtime_error(
                            "autodemo scheduler Reset did not complete");
                    }
                }

                write_autodemo_record(
                    autodemo_state.log,
                    autodemo_state.render_frame,
                    autodemo_state.runtime_tick,
                    interaction_scheduler.phase(),
                    interaction_output,
                    autodemo_carry_command_frame,
                    displayed_root,
                    root_displacement_m,
                    autodemo_evidence_capture.value(),
                    autodemo_action);

                if (!autodemo_state.carry_origin_captured &&
                    autodemo_state.render_frame >=
                        kAutodemoCarryDeadlineFrames)
                {
                    throw std::runtime_error(
                        "autodemo did not reach Carry by its 25 Hz deadline");
                }

                if (autodemo_action == AutodemoAction::Reset)
                {
                    if (autodemo_state.collapsed_states.size() != 7U ||
                        !autodemo_state.candidate_verified ||
                        autodemo_state.carry_command_count !=
                            kAutodemoCarryCommandCount ||
                        !autodemo_state.screenshot_captured ||
                        !(autodemo_state.last_carry_displacement_m > 0.20F))
                    {
                        throw std::runtime_error(
                            "autodemo final evidence contract was not met");
                    }
                    autodemo_state.log.flush();
                    if (!autodemo_state.log)
                    {
                        throw std::runtime_error(
                            "cannot flush autodemo JSONL");
                    }
                    autodemo_state.log.close();
                    if (autodemo_state.log.fail())
                    {
                        throw std::runtime_error(
                            "cannot close autodemo JSONL");
                    }
                    autodemo_state.reset_pending = false;
                    autodemo_state.reset_presentation_frames_remaining =
                        kAutodemoResetPresentationFrames;
                }
                else
                {
                    ++autodemo_state.render_frame;
                    if (autodemo_state.render_frame >=
                        kAutodemoMaximumEvidenceFrames)
                    {
                        throw std::runtime_error(
                            "autodemo evidence exceeded its 25 Hz bound");
                    }
                }
            }
        }
        else if (placement_autodemo_enabled)
        {
            ++autodemo_state.warmup_render_ticks;
            const interaction::RuntimeState runtime_state =
                interaction_output.diagnostics.state;
            const vec3 displayed_root = bone_positions(0);
            const AutodemoEvidenceCapture& capture =
                autodemo_evidence_capture.value();
            if (!placement_autodemo_state.evidence_started)
            {
                if (runtime_state == interaction::RuntimeState::Locomotion)
                {
                    if (autodemo_state.runtime_tick == 0U)
                    {
                        throw std::runtime_error(
                            "placement auto-demo did not warm on a 25 Hz tick");
                    }
                    placement_autodemo_state.walking_origin = displayed_root;
                    placement_autodemo_state.previous_displayed_root =
                        displayed_root;
                    placement_autodemo_state.origin_captured = true;
                    placement_autodemo_state.initial_pickup_distance_m =
                        autodemo_planar_distance(
                            displayed_root, capture.object_world.position);
                    if (!(placement_autodemo_state
                              .initial_pickup_distance_m >
                          kPlacementAutodemoInitialDistanceM))
                    {
                        throw std::runtime_error(
                            "placement auto-demo default spawn must be more "
                            "than 2.80 m from the pickup object");
                    }
                    placement_autodemo_state.evidence_started = true;
                    placement_autodemo_state.attachment_initialized = true;
                    placement_autodemo_state.last_attached =
                        interaction_output.diagnostics.attached;
                }
                else if (runtime_state !=
                         interaction::RuntimeState::Disabled)
                {
                    throw std::runtime_error(
                        "placement auto-demo warmup produced an unexpected "
                        "state");
                }
                else if (autodemo_state.warmup_render_ticks >=
                         kAutodemoWarmupFrames)
                {
                    throw std::runtime_error(
                        "placement auto-demo warmup exceeded its 25 Hz "
                        "deadline");
                }
            }

            if (placement_autodemo_state.evidence_started)
            {
                validate_placement_autodemo_state_progression(
                    placement_autodemo_state,
                    interaction_output.diagnostics);
                const interaction::ResultCode result =
                    interaction_output.diagnostics.result;
                if (placement_autodemo_state.interact_pulsed &&
                    (runtime_state == interaction::RuntimeState::Disabled ||
                     result == interaction::ResultCode::Rejected ||
                     result == interaction::ResultCode::Cancelled ||
                     result == interaction::ResultCode::Failed ||
                     result == interaction::ResultCode::Reset))
                {
                    std::string message =
                        "placement auto-demo runtime failed after Interact: ";
                    message += "state=";
                    message += interaction::debug_draw::state_name(
                        runtime_state);
                    message += " result=";
                    message += interaction::debug_draw::result_name(result);
                    message += " reason=";
                    message += interaction::debug_draw::reason_name(
                        interaction_output.diagnostics.reason);
                    throw std::runtime_error(message);
                }
                if (!placement_autodemo_state.place_observed &&
                    interaction_surface_registry.find(
                        interaction_destination_surface_handle) == nullptr)
                {
                    placement_autodemo_state
                        .destination_generation_changed_before_place = true;
                    throw std::runtime_error(
                        "placement auto-demo destination generation changed "
                        "before Place");
                }

                const float displayed_root_speed_mps =
                    placement_autodemo_state.render_frame == 0U
                    ? 0.0F
                    : 25.0F * autodemo_planar_distance(
                          displayed_root,
                          placement_autodemo_state.previous_displayed_root);
                placement_autodemo_state.last_root_speed_mps =
                    displayed_root_speed_mps;
                const float walking_displacement_m =
                    autodemo_planar_distance(
                        displayed_root,
                        placement_autodemo_state.walking_origin);
                const float pickup_distance_m = autodemo_planar_distance(
                    displayed_root, capture.object_world.position);
                const bool in_standoff_band =
                    pickup_distance_m >=
                        kPlacementAutodemoArrivalConfig.minimum_standoff_m &&
                    pickup_distance_m <=
                        kPlacementAutodemoArrivalConfig.maximum_standoff_m;
                const float reach_position_error_m =
                    autodemo_planar_distance(
                        displayed_root,
                        placement_autodemo_state.reach_waypoint.position);
                const float interaction_position_error_m =
                    autodemo_planar_distance(
                        displayed_root,
                        placement_autodemo_state
                            .interaction_waypoint.position);
                const float reach_yaw_error_radians = std::abs(
                    autodemo_shortest_angle(
                        autodemo_yaw_radians(bone_rotations(0)) -
                        autodemo_yaw_radians(
                            placement_autodemo_state
                                .reach_waypoint.rotation)));

                if (interaction_output.diagnostics.attached !=
                    placement_autodemo_state.last_attached)
                {
                    ++placement_autodemo_state.attachment_transition_count;
                    if (placement_autodemo_state.last_attached &&
                        !interaction_output.diagnostics.attached)
                    {
                        ++placement_autodemo_state
                              .attached_to_free_transition_count;
                    }
                    placement_autodemo_state.last_attached =
                        interaction_output.diagnostics.attached;
                }

                const interaction::InteractionTarget* observed_pick_target =
                    interaction_registry.find_by_id(
                        interaction_authored_target.handle.id);
                if (runtime_state == interaction::RuntimeState::Align &&
                    observed_pick_target != nullptr &&
                    observed_pick_target->state == interaction::ObjectState::Targeted &&
                    observed_pick_target->owner_request + 1U ==
                        interaction_next_request_id &&
                    interaction_output.diagnostics.target ==
                        observed_pick_target->handle &&
                    interaction_output.diagnostics.object_state ==
                        interaction::ObjectState::Targeted &&
                    placement_autodemo_state
                            .pick_reservation_transition_count == 0U)
                {
                    ++placement_autodemo_state
                          .pick_reservation_transition_count;
                }

                const bool pickup_candidate_state =
                    runtime_state == interaction::RuntimeState::Align ||
                    runtime_state ==
                        interaction::RuntimeState::PickupReplay ||
                    runtime_state == interaction::RuntimeState::Hold ||
                    runtime_state == interaction::RuntimeState::Carry;
                if (pickup_candidate_state &&
                    interaction_output.diagnostics.clip != 0)
                {
                    throw std::runtime_error(
                        "placement auto-demo pickup accepted a non-clip-0 "
                        "candidate");
                }
                if (runtime_state == interaction::RuntimeState::Carry &&
                    !interaction_output.diagnostics.attached)
                {
                    throw std::runtime_error(
                        "placement auto-demo Carry lost attachment");
                }
                const bool owned_pre_release =
                    runtime_state == interaction::RuntimeState::Carry ||
                    runtime_state ==
                        interaction::RuntimeState::PlacePreflight ||
                    runtime_state ==
                        interaction::RuntimeState::PlaceAlign ||
                    runtime_state ==
                        interaction::RuntimeState::PlaceReplay;
                if (owned_pre_release &&
                    (interaction_output.diagnostics.object_state !=
                         interaction::ObjectState::Held ||
                     !interaction_output.diagnostics.attached ||
                     !interaction_output.owns_pose ||
                     !capture.grasp_evidence_valid ||
                     capture.hand_constraint_weight != 1.0F ||
                     !capture.hand_constraint_validated ||
                     !capture.hand_constraint_result.applied ||
                     !capture.hand_constraint_result.reachable))
                {
                    std::ostringstream message;
                    message
                        << "placement auto-demo lost its validated full-weight "
                           "rendered grasp"
                        << " state="
                        << interaction::debug_draw::state_name(runtime_state)
                        << " object_state="
                        << interaction::debug_draw::object_state_name(
                               interaction_output.diagnostics.object_state)
                        << " attached="
                        << (interaction_output.diagnostics.attached
                                ? "true" : "false")
                        << " owns_pose="
                        << (interaction_output.owns_pose ? "true" : "false")
                        << " grasp_valid="
                        << (capture.grasp_evidence_valid ? "true" : "false")
                        << " weight=" << capture.hand_constraint_weight
                        << " validated="
                        << (capture.hand_constraint_validated
                                ? "true" : "false")
                        << " applied="
                        << (capture.hand_constraint_result.applied
                                ? "true" : "false")
                        << " reachable="
                        << (capture.hand_constraint_result.reachable
                                ? "true" : "false")
                        << " reach_shortfall="
                        << capture.hand_constraint_result.reach_shortfall_m
                        << " target="
                        << interaction_output.diagnostics.target.id << ':'
                        << interaction_output.diagnostics.target.generation
                        << " affordance="
                        << interaction_output.diagnostics.affordance_id
                        << " hand="
                        << (interaction_output.diagnostics.hand ==
                                    interaction::Hand::Left
                                ? "left" : "right");
                    throw std::runtime_error(message.str());
                }

                if (placement_autodemo_state.interact_pulsed &&
                    !placement_autodemo_state.destination_latched &&
                    runtime_state != interaction::RuntimeState::Carry)
                {
                    ++placement_autodemo_state.pickup_to_carry_ticks;
                    if (placement_autodemo_state.pickup_to_carry_ticks >
                        kPlacementAutodemoPickupToCarryTicks)
                    {
                        throw std::runtime_error(
                            "placement auto-demo pickup-to-Carry exceeded "
                            "375 ticks");
                    }
                }

                bool far_preview_this_tick = false;
                if (placement_preview_called_this_tick)
                {
                    if (!placement_autodemo_state.live_preview.has_value())
                    {
                        throw std::runtime_error(
                            "placement auto-demo preview call produced no "
                            "live preview");
                    }
                    const interaction::PlaceStagingPreview& preview =
                        *placement_autodemo_state.live_preview;
                    if (!preview.accepted ||
                        preview.candidate.selection_id == 0U ||
                        preview.candidate.source_id == 0U ||
                        preview.ik_config_fingerprint == 0U ||
                        !autodemo_same_ik(preview.ik, interaction_config.ik) ||
                        !autodemo_same_ik(
                            preview.candidate.ik, interaction_config.ik))
                    {
                        std::ostringstream message;
                        message
                            << "placement auto-demo preview was not a "
                               "certified runtime-owned candidate: accepted="
                            << (preview.accepted ? "true" : "false")
                            << " reason="
                            << interaction::debug_draw::reason_name(
                                   preview.reason)
                            << " source_id="
                            << preview.candidate.source_id
                            << " selection_id="
                            << preview.candidate.selection_id
                            << " ik_fingerprint="
                            << preview.ik_config_fingerprint
                            << " preview_ik_exact="
                            << (autodemo_same_ik(
                                    preview.ik, interaction_config.ik)
                                    ? "true" : "false")
                            << " candidate_ik_exact="
                            << (autodemo_same_ik(
                                    preview.candidate.ik,
                                    interaction_config.ik)
                                    ? "true" : "false");
                        throw std::runtime_error(message.str());
                    }
                    placement_autodemo_state.current_selection_id =
                        preview.candidate.selection_id;
                    if (!placement_autodemo_state.initial_preview_verified)
                    {
                        if (runtime_state !=
                                interaction::RuntimeState::Carry ||
                            autodemo_action !=
                                AutodemoAction::LatchDestination ||
                            preview.ready ||
                            (preview.root_error_m <=
                                 interaction::kPlaceStagingMaximumRootErrorM &&
                             preview.yaw_error_radians <=
                                 interaction::
                                     kPlaceStagingMaximumYawErrorRadians))
                        {
                            throw std::runtime_error(
                                "placement auto-demo initial far preview was "
                                "not intrinsically not-ready");
                        }
                        placement_autodemo_state.initial_far_selection_id =
                            preview.candidate.selection_id;
                        placement_autodemo_state.preview_ik_fingerprint =
                            preview.ik_config_fingerprint;
                        placement_autodemo_state.carry_origin_root =
                            displayed_root;
                        placement_autodemo_state.carry_origin_object =
                            capture.object_world.position;
                        placement_autodemo_state.initial_preview_verified =
                            true;
                        far_preview_this_tick = true;
                    }
                    else if (preview.ik_config_fingerprint !=
                             placement_autodemo_state
                                 .preview_ik_fingerprint)
                    {
                        throw std::runtime_error(
                            "placement auto-demo preview IK fingerprint "
                            "changed during staging");
                    }

                    if (runtime_state ==
                        interaction::RuntimeState::PlacePreflight)
                    {
                        if (autodemo_action != AutodemoAction::Place ||
                            !preview.ready ||
                            preview.root_error_m >
                                interaction::
                                    kPlaceStagingMaximumRootErrorM ||
                            preview.yaw_error_radians >
                                interaction::
                                    kPlaceStagingMaximumYawErrorRadians ||
                            placement_autodemo_state
                                    .submitted_selection_id !=
                                preview.candidate.selection_id ||
                            preview.candidate.selection_id ==
                                placement_autodemo_state
                                    .initial_far_selection_id)
                        {
                            throw std::runtime_error(
                                "placement auto-demo Place edge did not use "
                                "the current distinct ready live preview");
                        }
                        if (placement_autodemo_state.carry_staging_ticks <
                                kPlacementAutodemoMinimumCarryTicks ||
                            placement_autodemo_state.carry_staging_ticks >
                                kPlacementAutodemoMaximumCarryTicks ||
                            autodemo_planar_distance(
                                displayed_root,
                                placement_autodemo_state
                                    .carry_origin_root) <= 0.20F ||
                            autodemo_planar_distance(
                                capture.object_world.position,
                                placement_autodemo_state
                                    .carry_origin_object) <= 0.20F)
                        {
                            throw std::runtime_error(
                                "placement auto-demo ordinary Carry staging "
                                "contract was not met");
                        }
                        placement_autodemo_state.place_observed = true;
                    }
                }

                const char* demo_phase = "pickup";
                uint32_t phase_counter =
                    placement_autodemo_state.pickup_to_carry_ticks;
                int carry_staging_tick = -1;
                if (autodemo_action == AutodemoAction::Approach)
                {
                    ++placement_autodemo_state.walk_ticks;
                    placement_autodemo_state.settle_ticks = 0U;
                    demo_phase = "approach";
                    phase_counter = placement_autodemo_state.walk_ticks;
                }
                else if (autodemo_action == AutodemoAction::Interact)
                {
                    demo_phase = "interact";
                    phase_counter = placement_autodemo_state.settle_ticks;
                }
                else if (!placement_autodemo_state.interact_pulsed &&
                         runtime_state ==
                             interaction::RuntimeState::Locomotion)
                {
                    const bool walk_contract_met =
                        placement_autodemo_state.walk_ticks >=
                            kPlacementAutodemoMinimumWalkTicks &&
                        walking_displacement_m >=
                            kPlacementAutodemoMinimumWalkDisplacementM;
                    if (placement_autodemo_state
                            .frozen_pick_entry_index.has_value() &&
                        walk_contract_met && in_standoff_band &&
                        displayed_root_speed_mps <=
                            kPlacementAutodemoSettleMaximumSpeedMps &&
                        interaction_position_error_m <=
                            kPlacementAutodemoAlignmentPositionErrorM &&
                        reach_yaw_error_radians <=
                            kPlacementAutodemoReachYawErrorRadians)
                    {
                        ++placement_autodemo_state.settle_ticks;
                    }
                    else
                    {
                        placement_autodemo_state.settle_ticks = 0U;
                    }
                    demo_phase = "settle";
                    phase_counter =
                        placement_autodemo_state.settle_ticks;
                }
                else if (runtime_state ==
                         interaction::RuntimeState::Carry)
                {
                    if (far_preview_this_tick)
                    {
                        demo_phase = "carry_preview";
                        phase_counter = 1U;
                    }
                    else if (placement_preview_called_this_tick)
                    {
                        if (autodemo_action != AutodemoAction::Stage ||
                            placement_autodemo_state.live_preview->ready ||
                            length(placement_autodemo_state
                                       .left_stick_command) <= 1.0e-4F)
                        {
                            throw std::runtime_error(
                                "placement auto-demo staging escaped "
                                "ordinary nonzero Carry input");
                        }
                        carry_staging_tick = static_cast<int>(
                            placement_autodemo_state.carry_staging_ticks);
                        ++placement_autodemo_state.carry_staging_ticks;
                        if (placement_autodemo_state.carry_staging_ticks >
                            kPlacementAutodemoMaximumCarryTicks)
                        {
                            throw std::runtime_error(
                                "placement auto-demo Carry staging exceeded "
                                "150 ticks");
                        }
                        demo_phase = "carry_staging";
                        phase_counter =
                            placement_autodemo_state.carry_staging_ticks;
                    }
                }

                const bool place_runtime_state =
                    runtime_state ==
                        interaction::RuntimeState::PlacePreflight ||
                    runtime_state ==
                        interaction::RuntimeState::PlaceAlign ||
                    runtime_state ==
                        interaction::RuntimeState::PlaceReplay ||
                    runtime_state ==
                        interaction::RuntimeState::PlaceRelease;
                if (place_runtime_state)
                {
                    ++placement_autodemo_state.place_ticks;
                    if (placement_autodemo_state.place_ticks >
                        kPlacementAutodemoPlaceTicks)
                    {
                        throw std::runtime_error(
                            "placement auto-demo Place exceeded 250 ticks");
                    }
                    demo_phase = "place";
                    phase_counter = placement_autodemo_state.place_ticks;
                }

                if (runtime_state ==
                        interaction::RuntimeState::PlaceRelease &&
                    interaction_output.diagnostics.place.released &&
                    !placement_autodemo_state.actual_fit_captured)
                {
                    placement_autodemo_state.final_actual_fit =
                        interaction_output.diagnostics.place.actual_fit;
                    placement_autodemo_state.final_support_sweep_clear =
                        interaction_output.diagnostics.place
                            .support_sweep_clear;
                    placement_autodemo_state.actual_fit_captured = true;
                }

                const interaction::InteractionTarget* committed_target =
                    interaction_registry.find_by_id(
                        interaction_authored_target.handle.id);
                if (committed_target != nullptr &&
                    committed_target->state ==
                        interaction::ObjectState::Free &&
                    placement_autodemo_state.place_observed)
                {
                    placement_autodemo_state.destination_support_committed =
                        length(
                            committed_target->table_world.position -
                            interaction_authored_destination_surface
                                .support_volume_world.position) <= 1.0e-5F &&
                        quat_angle_between(
                            committed_target->table_world.rotation,
                            interaction_authored_destination_surface
                                .support_volume_world.rotation) <= 1.0e-5F &&
                        length(
                            committed_target->table_size -
                            interaction_authored_destination_surface
                                .support_volume_size) <= 1.0e-5F;
                }

                const bool final_locomotion =
                    placement_autodemo_state.place_observed &&
                    runtime_state ==
                        interaction::RuntimeState::Locomotion;
                if (final_locomotion)
                {
                    ++placement_autodemo_state.handoff_frames_recorded;
                    if (placement_autodemo_state.handoff_frames_recorded >
                        kPlacementAutodemoMaximumHandoffFrames)
                    {
                        throw std::runtime_error(
                            "placement auto-demo handoff exceeded seventy-five "
                            "frames");
                    }
                    if (!interaction_frame_state.overrides_locomotion_pose)
                    {
                        ++placement_autodemo_state
                              .post_release_settled_frames_recorded;
                    }
                    else
                    {
                        placement_autodemo_state
                            .post_release_settled_frames_recorded = 0U;
                    }
                    const bool placement_presentation_ready =
                        placement_autodemo_state.handoff_frames_recorded >=
                            kPlacementAutodemoHandoffFrames &&
                        placement_autodemo_state
                                .post_release_settled_frames_recorded >=
                            kPlacementAutodemoSettledFrames;
                    placement_autodemo_state.handoff_frames_remaining =
                        placement_presentation_ready ? 0U : 1U;
                    demo_phase = "handoff";
                    phase_counter =
                        placement_autodemo_state.handoff_frames_recorded;
                }

                const bool publish_pick_preview_failure =
                    placement_autodemo_state
                            .pending_pick_preview_failure !=
                        PlacementPickPreviewFailure::None &&
                    !placement_autodemo_state
                         .pick_preview_failure_recorded;
                std::optional<ControllerPlacementAutodemoState>
                    terminal_record_state;
                if (publish_pick_preview_failure)
                {
                    terminal_record_state = placement_autodemo_state;
                    terminal_record_state->pick_preview_failure_recorded =
                        true;
                }
                const ControllerPlacementAutodemoState& record_state =
                    terminal_record_state.has_value()
                    ? *terminal_record_state
                    : placement_autodemo_state;
                write_placement_autodemo_record(
                    autodemo_state.log,
                    record_state,
                    autodemo_state.runtime_tick,
                    interaction_scheduler.phase(),
                    interaction_output,
                    interaction_config,
                    interaction_authored_target,
                    interaction_authored_destination_surface,
                    interaction_destination_affordance_id,
                    interaction_authored_target.object_bounds,
                    capture,
                    displayed_root,
                    pickup_distance_m,
                    walking_displacement_m,
                    displayed_root_speed_mps,
                    autodemo_action,
                    demo_phase,
                    phase_counter,
                    carry_staging_tick,
                    placement_preview_called_this_tick,
                    placement_autodemo_state
                        .destination_support_committed);
                if (publish_pick_preview_failure)
                {
                    autodemo_state.log.flush();
                    if (!autodemo_state.log)
                    {
                        throw std::runtime_error(
                            "cannot flush terminal pickup preview evidence");
                    }
                    placement_autodemo_state
                        .pick_preview_failure_recorded = true;
                }

                placement_autodemo_state.previous_displayed_root =
                    displayed_root;
                if (placement_autodemo_state.walk_ticks >
                        kPlacementAutodemoMaximumWalkTicks &&
                    !placement_autodemo_state.interact_pulsed &&
                    !placement_autodemo_state.reach_entry_reached)
                {
                    std::string message =
                        "placement auto-demo approach exceeded 250 ticks: ";
                    message += "entry_reached=";
                    message += placement_autodemo_state.reach_entry_reached
                        ? "true" : "false";
                    message += " entry_error_m=" + std::to_string(
                        autodemo_planar_distance(
                            displayed_root,
                            placement_autodemo_state.reach_entry_point));
                    message += " waypoint_position_error_m=" +
                        std::to_string(reach_position_error_m);
                    message += " waypoint_yaw_error_degrees=" +
                        std::to_string(
                            reach_yaw_error_radians * 180.0F / PIf);
                    message += " pickup_distance_m=" +
                        std::to_string(pickup_distance_m);
                    message += " root_speed_mps=" +
                        std::to_string(displayed_root_speed_mps);
                    message += " walk_displacement_m=" +
                        std::to_string(walking_displacement_m);
                    throw std::runtime_error(message);
                }

                if (final_locomotion)
                {
                    const interaction::PlaceAffordance& place_affordance =
                        interaction_authored_destination_surface
                            .affordances.front();
                    const interaction::Transform place_goal =
                        interaction::placement_goal_world(
                            interaction_authored_destination_surface,
                            place_affordance.object_in_surface);
                    const float final_orientation_error_degrees =
                        quat_angle_between(
                            capture.object_world.rotation,
                            place_goal.rotation) * 180.0F / PIf;
                    const interaction::PlaceMotionMode mode =
                        placement_autodemo_state.live_preview.has_value()
                        ? placement_autodemo_state.live_preview
                              ->candidate.mode
                        : interaction::PlaceMotionMode::None;
                    const bool instrumentation_clean =
                        placement_autodemo_state
                                .locomotion_provider_calls ==
                            placement_autodemo_state
                                .live_flat_provider_calls &&
                        placement_autodemo_state
                                .canonical_snapshot_provider_calls == 0U &&
                        placement_autodemo_state.root_relocation_calls == 0U &&
                        placement_autodemo_state
                                .simulation_root_initialization_calls == 0U &&
                        placement_autodemo_state
                                .displayed_root_initialization_calls == 0U &&
                        placement_autodemo_state.pick_resolver_calls == 1U &&
                        placement_autodemo_state
                            .pick_resolver_used_live_flat &&
                        placement_autodemo_state
                                .placement_surface_resolver_calls == 0U &&
                        placement_autodemo_state.free_selector_calls == 0U &&
                        placement_autodemo_state
                                .external_match_input_calls == 0U &&
                        placement_autodemo_state.preview_mutation_count == 0U &&
                        placement_autodemo_state.direct_root_write_calls == 0U &&
                        !placement_autodemo_state
                             .destination_generation_changed_before_place;
                    if (interaction_output.diagnostics.result !=
                            interaction::ResultCode::Succeeded ||
                        interaction_output.diagnostics.reason !=
                            interaction::Reason::None ||
                        interaction_output.diagnostics.object_state !=
                            interaction::ObjectState::Free ||
                        interaction_output.diagnostics.attached ||
                        interaction_output.owns_pose ||
                        placement_autodemo_state.collapsed_states.size() !=
                            11U ||
                        placement_autodemo_state.walk_ticks <
                            kPlacementAutodemoMinimumWalkTicks ||
                        walking_displacement_m <
                            kPlacementAutodemoMinimumWalkDisplacementM ||
                        placement_autodemo_state.settle_ticks <
                            kPlacementAutodemoSettleTicks ||
                        placement_autodemo_state
                                .attachment_transition_count != 2U ||
                        placement_autodemo_state
                                .attached_to_free_transition_count != 1U ||
                        !placement_autodemo_state.actual_fit_captured ||
                        !placement_autodemo_state.final_actual_fit.accepted ||
                        !placement_autodemo_state.final_actual_fit
                             .footprint_valid ||
                        !placement_autodemo_state.final_actual_fit
                             .overhead_valid ||
                        placement_autodemo_state.final_actual_fit
                                .support_gap_m < -0.005F ||
                        placement_autodemo_state.final_actual_fit
                                .support_gap_m > 0.020F ||
                        !placement_autodemo_state
                             .final_support_sweep_clear ||
                        !placement_autodemo_state
                             .destination_support_committed ||
                        length(
                            capture.object_world.position -
                            place_goal.position) > 0.020F ||
                        final_orientation_error_degrees > 10.0F ||
                        (mode !=
                             interaction::PlaceMotionMode::RecordedPlace &&
                         mode != interaction::PlaceMotionMode::
                                     ReversedPickup) ||
                        !instrumentation_clean)
                    {
                        throw std::runtime_error(
                            "placement auto-demo final evidence contract was "
                            "not met");
                    }
                    if (placement_autodemo_state.handoff_frames_recorded >=
                            kPlacementAutodemoHandoffFrames &&
                        placement_autodemo_state
                                .post_release_settled_frames_recorded >=
                            kPlacementAutodemoSettledFrames)
                    {
                        const std::string screenshot_temporary =
                            autodemo_configuration->screenshot_temporary
                                .string();
                        TakeScreenshot(screenshot_temporary.c_str());
                        validate_autodemo_screenshot(
                            *autodemo_configuration);
                        placement_autodemo_state.screenshot_captured = true;
                        autodemo_state.log.flush();
                        if (!autodemo_state.log)
                        {
                            throw std::runtime_error(
                                "cannot flush placement auto-demo JSONL");
                        }
                        autodemo_state.log.close();
                        if (autodemo_state.log.fail())
                        {
                            throw std::runtime_error(
                                "cannot close placement auto-demo JSONL");
                        }
                        placement_autodemo_state.final_record_written = true;
                        publish_autodemo_evidence(
                            *autodemo_configuration);
                        autodemo_state.complete = true;
                        autodemo_state.exit_requested = true;
                    }
                    else
                    {
                        ++placement_autodemo_state.render_frame;
                        if (placement_autodemo_state.render_frame >=
                            kPlacementAutodemoMaximumEvidenceFrames)
                        {
                            throw std::runtime_error(
                                "placement auto-demo evidence exceeded its "
                                "900 record bound");
                        }
                    }
                }
                else
                {
                    ++placement_autodemo_state.render_frame;
                    if (placement_autodemo_state.render_frame >=
                        kPlacementAutodemoMaximumEvidenceFrames)
                    {
                        throw std::runtime_error(
                            "placement auto-demo evidence exceeded its 900 "
                            "record bound");
                    }
                }
            }
        }

        ++smart_pickup_controller_tick;

    };

#if defined(PLATFORM_WEB)
    std::function<void()> u{update_func};
    emscripten_set_main_loop_arg(update_callback, &u, 25, 1);
#else
    while (!autodemo_state.exit_requested)
    {
        if (autodemo_configuration.has_value() &&
            autodemo_sigterm_requested != 0)
        {
            autodemo_state.failure = "autodemo received SIGTERM";
            autodemo_state.exit_requested = true;
            break;
        }
        if (WindowShouldClose())
        {
            if (autodemo_configuration.has_value() &&
                !autodemo_state.complete)
            {
                autodemo_state.failure =
                    "autodemo window closed before successful Reset";
            }
            break;
        }
        if (!autodemo_configuration.has_value())
        {
            update_func();
            continue;
        }
        try
        {
            update_func();
        }
        catch (const std::exception& error)
        {
            autodemo_state.failure = error.what();
            autodemo_state.exit_requested = true;
        }
        catch (...)
        {
            autodemo_state.failure = "autodemo failed with unknown exception";
            autodemo_state.exit_requested = true;
        }
    }
#endif

    int exit_code = 0;
    if (autodemo_configuration.has_value() && !autodemo_state.complete)
    {
        if (autodemo_state.failure.empty())
        {
            autodemo_state.failure =
                "autodemo exited before successful Reset";
        }
        if (autodemo_state.log.is_open())
        {
            autodemo_state.log.close();
        }
        cleanup_autodemo_temporaries(*autodemo_configuration);
        std::fprintf(
            stderr, "controller: %s\n", autodemo_state.failure.c_str());
        exit_code = 1;
    }

    // Unload stuff and finish (G1: no character mesh/shader to unload)
    UnloadModel(ground_plane_model);
    UnloadShader(ground_plane_shader);

    CloseWindow();

    return exit_code;
#if !defined(PLATFORM_WEB)
    }
    catch (const std::exception& error)
    {
        if (autodemo_configuration.has_value())
        {
            cleanup_autodemo_temporaries(*autodemo_configuration);
        }
        std::fprintf(stderr, "controller: %s\n", error.what());
        UnloadModel(ground_plane_model);
        UnloadShader(ground_plane_shader);
        CloseWindow();
        return 1;
    }
    catch (...)
    {
        if (autodemo_configuration.has_value())
        {
            cleanup_autodemo_temporaries(*autodemo_configuration);
        }
        std::fprintf(
            stderr, "controller: unknown interaction startup failure\n");
        UnloadModel(ground_plane_model);
        UnloadShader(ground_plane_shader);
        CloseWindow();
        return 1;
    }
#endif
}
