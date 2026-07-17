#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "sonic/cpp/mm_chunk_json.h"
#include "sonic/cpp/g1_database_validation.h"
#include "sonic/cpp/g1_joint_contract_io.h"
#include "sonic/cpp/g1_joint_feasibility.h"
#include "sonic/cpp/g1_joint_projection.h"
#include "sonic/cpp/g1_runtime.h"
#include "sonic/cpp/sonic_flat_scene.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#ifndef MM_CHUNK_BUILD_COMMIT
#define MM_CHUNK_BUILD_COMMIT "unknown"
#endif

#ifndef MM_CHUNK_DEFAULT_JOINT_CONTRACT
#define MM_CHUNK_DEFAULT_JOINT_CONTRACT "sonic/configs/g1_joint_contract.json"
#endif

#ifndef MM_CHUNK_DEFAULT_SCENE_REGISTRY
#define MM_CHUNK_DEFAULT_SCENE_REGISTRY "sonic/configs/scene_registry.json"
#endif

static constexpr const char* MM_CHUNK_SCHEMA = "mm-chunk/v1";
static constexpr const char* MM_CHUNK_JOINT_FEASIBILITY_SCHEMA =
    "g1-joint-feasibility-certificate/v1";

struct mm_server_joint_feasibility_identity
{
    int frame_count = 0;
    int raw_safe_count = 0;
    int raw_unsafe_count = 0;
    int search_safe_count = 0;
    std::string mask_sha256;
    int joint_limit_violation_count[SonicG1JointCount] = {};
};

struct mm_server_identity
{
    std::vector<std::string> source_joint_names;
    std::vector<std::string> target_joint_names;
    std::string skeleton_signature;
    std::string build_commit;
    std::string joint_contract_sha256;
    std::string motion_manifest_sha256;
    std::string database_sha256;
    std::string terrain_features_sha256;
    std::string terrain_support_sha256;
    std::string scene_index_sha256;
    std::string coordinate_signature;
    mm_server_joint_feasibility_identity joint_feasibility;
};

struct mm_server_scene_identity
{
    std::string scene_id;
    std::string route_id;
    float terrain_weight = 0.0f;
    std::string coordinate_signature;
    std::string heightfield_sha256;
    std::string mesh_sha256;
    std::string walkability_sha256;
};

static const char* const mm_server_source_joint_names[MM_CHUNK_JOINT_COUNT] = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
};

static const char* const mm_server_target_joint_names[MM_CHUNK_JOINT_COUNT] = {
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
};

static void mm_server_fill_fixed_joint_names(mm_server_identity& identity)
{
    identity.source_joint_names.assign(
        mm_server_source_joint_names,
        mm_server_source_joint_names + MM_CHUNK_JOINT_COUNT);
    identity.target_joint_names.assign(
        mm_server_target_joint_names,
        mm_server_target_joint_names + MM_CHUNK_JOINT_COUNT);
}

static void mm_server_set_joint_feasibility_identity(
    mm_server_joint_feasibility_identity& output,
    const sonic_joint_feasibility_certificate& certificate)
{
    mm_server_joint_feasibility_identity candidate;
    candidate.frame_count = certificate.frame_count;
    candidate.raw_safe_count = certificate.raw_safe_count;
    candidate.raw_unsafe_count = certificate.raw_unsafe_count;
    candidate.search_safe_count = certificate.search_safe_count;
    candidate.mask_sha256 = certificate.mask_sha256;
    for (int joint = 0; joint < SonicG1JointCount; ++joint) {
        candidate.joint_limit_violation_count[joint] =
            certificate.joint_limit_violation_count[joint];
    }
    output = candidate;
}

struct mm_matching_feature_storage
{
    array2d<float> features;
    array1d<float> offset;
    array1d<float> scale;
    array2d<float> bound_sm_min;
    array2d<float> bound_sm_max;
    array2d<float> bound_lr_min;
    array2d<float> bound_lr_max;

    static void swap_array(array2d<float>& first, array2d<float>& second)
    {
        std::swap(first.rows, second.rows);
        std::swap(first.cols, second.cols);
        std::swap(first.data, second.data);
    }

    static void swap_array(array1d<float>& first, array1d<float>& second)
    {
        std::swap(first.size, second.size);
        std::swap(first.data, second.data);
    }

    void swap_with(database& db)
    {
        swap_array(features, db.features);
        swap_array(offset, db.features_offset);
        swap_array(scale, db.features_scale);
        swap_array(bound_sm_min, db.bound_sm_min);
        swap_array(bound_sm_max, db.bound_sm_max);
        swap_array(bound_lr_min, db.bound_lr_min);
        swap_array(bound_lr_max, db.bound_lr_max);
    }
};

struct mm_real_reset_context
{
    scene_pack scene;
    mm_matching_feature_storage features;
    mm_server_scene_identity identity;
};

class mm_real_adapter
{
public:
    using state_type = g1_controller_state;
    using reset_context_type = mm_real_reset_context;

    bool initialize(
        const char* terrain_root,
        const char* contract_path,
        const char* registry_path,
        std::string& message)
    {
        char error[1024] = {};
        terrain_root_ = terrain_root != nullptr ? terrain_root : "";
        identity_.build_commit = MM_CHUNK_BUILD_COMMIT;
        identity_.skeleton_signature = G1_SkeletonSignature;
        if (terrain_root_.empty() || contract_path == nullptr ||
            contract_path[0] == '\0' || registry_path == nullptr ||
            registry_path[0] == '\0') {
            message =
                "terrain root, joint contract, and scene registry are required";
            return false;
        }
        std::string manifest_path;
        sonic_joint_contract_metadata contract_metadata;
        if (!scene_join(
                manifest_path,
                terrain_root_.c_str(),
                "manifest.json",
                error,
                static_cast<int>(sizeof(error))) ||
            !motion_manifest_load_and_verify(
                manifest_,
                terrain_root_.c_str(),
                error,
                static_cast<int>(sizeof(error))) ||
            !sha256_file_hex(
                identity_.motion_manifest_sha256,
                manifest_path.c_str(),
                error,
                static_cast<int>(sizeof(error))) ||
            !sonic_joint_contract_load(
                contract_,
                &contract_metadata,
                contract_path,
                error,
                static_cast<int>(sizeof(error))) ||
            !sha256_file_hex(
                identity_.joint_contract_sha256,
                contract_path,
                error,
                static_cast<int>(sizeof(error))) ||
            !sonic_flat_scene_definition_load(
                flat_definition_,
                registry_path,
                error,
                static_cast<int>(sizeof(error)))) {
            message = error;
            return false;
        }
        mm_server_identity fixed_names;
        mm_server_fill_fixed_joint_names(fixed_names);
        if (contract_metadata.source_joint_names !=
                fixed_names.source_joint_names ||
            contract_metadata.target_joint_names !=
                fixed_names.target_joint_names) {
            message = "joint contract source/target order is not fixed";
            return false;
        }
        identity_.source_joint_names = contract_metadata.source_joint_names;
        identity_.target_joint_names = contract_metadata.target_joint_names;
        identity_.database_sha256 = manifest_.database.sha256;
        identity_.terrain_features_sha256 =
            manifest_.terrain_features.sha256;
        identity_.terrain_support_sha256 = manifest_.terrain_support.sha256;
        identity_.scene_index_sha256 = manifest_.scene_index.sha256;
        identity_.coordinate_signature = manifest_.surface.coordinate_signature;

        std::string database_path;
        std::string feature_path;
        std::string support_path;
        terrain_feature_set terrain_rows;
        if (!scene_join(
                database_path,
                terrain_root_.c_str(),
                manifest_.database.path,
                error,
                static_cast<int>(sizeof(error))) ||
            !scene_join(
                feature_path,
                terrain_root_.c_str(),
                manifest_.terrain_features.path,
                error,
                static_cast<int>(sizeof(error))) ||
            !scene_join(
                support_path,
                terrain_root_.c_str(),
                manifest_.terrain_support.path,
                error,
                static_cast<int>(sizeof(error))) ||
            !terrain_features_load(
                terrain_rows,
                feature_path.c_str(),
                error,
                static_cast<int>(sizeof(error)))) {
            message = error;
            return false;
        }

        database_load(database_, database_path.c_str());
        if (!g1_database_validate(
                database_, error, static_cast<int>(sizeof(error))) ||
            terrain_rows.values.rows != database_.nframes() ||
            terrain_rows.values.cols != 4) {
            message = error[0] != '\0'
                ? error
                : "database and terrain feature shapes differ";
            return false;
        }
        if (!sonic_build_joint_feasibility_certificate(
                joint_feasibility_,
                database_,
                contract_,
                error,
                static_cast<int>(sizeof(error)))) {
            message = error;
            return false;
        }
        mm_server_set_joint_feasibility_identity(
            identity_.joint_feasibility, joint_feasibility_);
        std::swap(database_.terrain_features.rows, terrain_rows.values.rows);
        std::swap(database_.terrain_features.cols, terrain_rows.values.cols);
        std::swap(database_.terrain_features.data, terrain_rows.values.data);
        if (!rebuild_matching_features(0.0f, error, sizeof(error)) ||
            !terrain_support_load(
                support_,
                support_path.c_str(),
                database_.nframes(),
                error,
                static_cast<int>(sizeof(error))) ||
            !scene_catalog_load(
                catalog_,
                terrain_root_.c_str(),
                manifest_,
                error,
                static_cast<int>(sizeof(error)))) {
            message = error;
            return false;
        }
        return true;
    }

    const mm_server_identity& identity() const { return identity_; }
    const mm_server_scene_identity& scene_identity() const
    {
        return scene_identity_;
    }

    const mm_server_scene_identity& prepared_scene_identity(
        const reset_context_type& context) const
    {
        return context.identity;
    }

    bool prepare_reset(
        state_type& output,
        mm_chunk_boundary& boundary,
        reset_context_type& context,
        const mm_chunk_reset_request& request,
        std::string& message)
    {
        char error[1024] = {};
        if (request.scene_id == SonicFlatSceneId) {
            if (!sonic_flat_scene_build(
                    context.scene,
                    flat_definition_,
                    request.route_id,
                    error,
                    static_cast<int>(sizeof(error)))) {
                message = error;
                return false;
            }
        } else {
            const int scene_index =
                scene_catalog_find(catalog_, request.scene_id.c_str());
            if (scene_index < 0 ||
                static_cast<std::size_t>(scene_index) >= catalog_.ids.size() ||
                catalog_.ids[static_cast<std::size_t>(scene_index)] !=
                    request.scene_id) {
                message = "unknown scene: " + request.scene_id;
                return false;
            }
            if (!scene_pack_load(
                    context.scene,
                    terrain_root_.c_str(),
                    manifest_,
                    catalog_,
                    scene_index,
                    error,
                    static_cast<int>(sizeof(error)))) {
                message = error;
                return false;
            }
        }
        const scene_route* route =
            scene_route_find(context.scene.metadata, request.route_id.c_str());
        if (route == nullptr || route->id != request.route_id) {
            message = "scene " + request.scene_id + " has no route " +
                      request.route_id;
            return false;
        }

        mm_matching_feature_storage prior_features;
        if (!rebuild_matching_features(
                request.terrain_weight,
                prior_features,
                error,
                sizeof(error))) {
            message = error;
            return false;
        }

        state_type next_state;
        if (!g1_controller_state_reset(
                next_state,
                database_,
                support_,
                context.scene,
                error,
                static_cast<int>(sizeof(error)))) {
            context.features.swap_with(database_);
            prior_features.swap_with(database_);
            message = error;
            return false;
        }
        next_state.route_index = static_cast<int>(
            route - context.scene.metadata.routes.data());
        next_state.route_waypoint = 1;
        next_state.route_frames = 0;
        next_state.adjusted_bone_rotations = next_state.bone_rotations;
        support_pose_apply(
            next_state.adjusted_bone_positions,
            next_state.bone_positions,
            next_state.support.height);
        forward_kinematics_full(
            next_state.global_bone_positions,
            next_state.global_bone_rotations,
            next_state.adjusted_bone_positions,
            next_state.adjusted_bone_rotations,
            database_.bone_parents);
        if (!observe_boundary(boundary, next_state, error, sizeof(error))) {
            context.features.swap_with(database_);
            prior_features.swap_with(database_);
            message = error;
            return false;
        }

        context.features.swap_with(database_);
        prior_features.swap_with(database_);
        g1_controller_state_swap(output, next_state);
        context.identity.scene_id = request.scene_id;
        context.identity.route_id = request.route_id;
        context.identity.terrain_weight = request.terrain_weight;
        context.identity.coordinate_signature =
            context.scene.metadata.coordinate_signature;
        context.identity.heightfield_sha256 =
            context.scene.metadata.heightfield.sha256;
        context.identity.mesh_sha256 = context.scene.metadata.mesh.sha256;
        context.identity.walkability_sha256 =
            context.scene.metadata.walkability.sha256;
        return true;
    }

    void publish_reset(reset_context_type& context)
    {
        context.features.swap_with(database_);
        scene_pack_swap(active_scene_, context.scene);
        using std::swap;
        swap(scene_identity_, context.identity);
    }

    bool clone(
        state_type& destination,
        const state_type& source,
        std::string& message)
    {
        char error[1024] = {};
        if (!g1_controller_state_clone(
                destination,
                source,
                error,
                static_cast<int>(sizeof(error)))) {
            message = error;
            return false;
        }
        return true;
    }

    void swap(state_type& first, state_type& second)
    {
        g1_controller_state_swap(first, second);
    }

    bool observe(
        mm_chunk_boundary& boundary,
        const state_type& state,
        std::string& message)
    {
        char error[1024] = {};
        if (!observe_boundary(boundary, state, error, sizeof(error))) {
            message = error;
            return false;
        }
        return true;
    }

    bool advance(
        mm_chunk_step_diagnostic& diagnostic,
        state_type& state,
        const mm_chunk_generate_request& request,
        int,
        std::string& message)
    {
        g1_runtime_step_request runtime_request;
        runtime_request.mode = G1RuntimeDirect;
        runtime_request.requested_velocity_holden = vec3(
            request.requested_velocity_holden[0],
            request.requested_velocity_holden[1],
            request.requested_velocity_holden[2]);
        runtime_request.desired_heading_holden = quat(
            request.desired_heading_holden_wxyz[0],
            request.desired_heading_holden_wxyz[1],
            request.desired_heading_holden_wxyz[2],
            request.desired_heading_holden_wxyz[3]);
        runtime_request.matching_enabled = true;
        g1_runtime_step_result result;
        g1_runtime_config config;
        g1_runtime_frame_feasibility runtime_feasibility;
        runtime_feasibility.raw_safe = joint_feasibility_.raw_safe.data;
        runtime_feasibility.search_safe = joint_feasibility_.search_safe.data;
        runtime_feasibility.count = joint_feasibility_.frame_count;
        char error[1024] = {};
        if (!g1_runtime_step(
                result,
                state,
                database_,
                support_,
                active_scene_,
                runtime_feasibility,
                runtime_request,
                config,
                error,
                static_cast<int>(sizeof(error)))) {
            message = error;
            return false;
        }
        diagnostic = mm_chunk_step_diagnostic();
        diagnostic.selected_database_frame = result.selected_database_frame;
        diagnostic.searched = state.searched;
        diagnostic.transitioned = state.transitioned;
        diagnostic.terrain_cost = state.selected_terrain_error;
        diagnostic.applied_velocity_holden[0] =
            state.command.applied_velocity.x;
        diagnostic.applied_velocity_holden[1] =
            state.command.applied_velocity.y;
        diagnostic.applied_velocity_holden[2] =
            state.command.applied_velocity.z;
        diagnostic.support_height = state.support.height;
        diagnostic.support_target = state.support.nominal_height;
        for (int sample = 0; sample < MM_CHUNK_TERRAIN_SAMPLE_COUNT;
             ++sample) {
            diagnostic.terrain_values[sample] = result.terrain.values[sample];
            diagnostic.terrain_points_holden[sample][0] =
                result.terrain.points[sample].x;
            diagnostic.terrain_points_holden[sample][1] =
                result.terrain.points[sample].y;
            diagnostic.terrain_points_holden[sample][2] =
                result.terrain.points[sample].z;
        }
        return true;
    }

private:
    bool rebuild_matching_features(
        float terrain_weight,
        mm_matching_feature_storage& prior,
        char* error,
        int capacity)
    {
        prior.swap_with(database_);
        database_build_matching_features(
            database_,
            0.75f,
            1.0f,
            1.0f,
            1.0f,
            1.5f,
            G1_LeftAnkle,
            G1_RightAnkle,
            G1_Hips,
            terrain_weight);
        if (!g1_matching_features_validate(
                database_, error, capacity) ||
            !motion_manifest_validate_database(
                manifest_, database_, error, capacity)) {
            mm_matching_feature_storage failed;
            failed.swap_with(database_);
            prior.swap_with(database_);
            return false;
        }
        return true;
    }

    bool rebuild_matching_features(
        float terrain_weight,
        char* error,
        int capacity)
    {
        mm_matching_feature_storage prior;
        return rebuild_matching_features(
            terrain_weight, prior, error, capacity);
    }

    bool observe_boundary(
        mm_chunk_boundary& output,
        const state_type& state,
        char* error,
        int capacity) const
    {
        sonic_projected_pose projection;
        if (!sonic_project_pose(
                projection,
                contract_,
                state.bone_rotations,
                state.bone_angular_velocities,
                state.global_bone_positions,
                state.global_bone_rotations,
                error,
                capacity)) {
            return false;
        }
        mm_chunk_boundary candidate;
        for (int joint = 0; joint < MM_CHUNK_JOINT_COUNT; ++joint) {
            candidate.joint_position_source[joint] =
                projection.source_joint_position[joint];
            candidate.joint_velocity_source[joint] =
                projection.source_joint_velocity[joint];
        }
        candidate.physical_pelvis_position_holden[0] =
            projection.physical_pelvis_position_holden.x;
        candidate.physical_pelvis_position_holden[1] =
            projection.physical_pelvis_position_holden.y;
        candidate.physical_pelvis_position_holden[2] =
            projection.physical_pelvis_position_holden.z;
        candidate.physical_pelvis_orientation_holden[0] =
            projection.physical_pelvis_orientation_holden.w;
        candidate.physical_pelvis_orientation_holden[1] =
            projection.physical_pelvis_orientation_holden.x;
        candidate.physical_pelvis_orientation_holden[2] =
            projection.physical_pelvis_orientation_holden.y;
        candidate.physical_pelvis_orientation_holden[3] =
            projection.physical_pelvis_orientation_holden.z;
        candidate.virtual_root_position_holden[0] = state.simulation_position.x;
        candidate.virtual_root_position_holden[1] = state.simulation_position.y;
        candidate.virtual_root_position_holden[2] = state.simulation_position.z;
        const quat virtual_rotation =
            sonic_projection_quat_canonical(state.simulation_rotation);
        candidate.virtual_root_orientation_holden[0] = virtual_rotation.w;
        candidate.virtual_root_orientation_holden[1] = virtual_rotation.x;
        candidate.virtual_root_orientation_holden[2] = virtual_rotation.y;
        candidate.virtual_root_orientation_holden[3] = virtual_rotation.z;
        output = candidate;
        return true;
    }

    std::string terrain_root_;
    sonic_flat_scene_definition flat_definition_;
    mm_server_identity identity_;
    mm_server_scene_identity scene_identity_;
    motion_pack_manifest manifest_;
    scene_catalog catalog_;
    scene_pack active_scene_;
    database database_;
    terrain_support_set support_;
    sonic_joint_contract_entry contract_[SonicG1JointCount];
    sonic_joint_feasibility_certificate joint_feasibility_;
};

struct mm_fake_state
{
    int frame = 0;
    std::vector<int> history;
};

struct mm_fake_reset_context
{
    mm_server_scene_identity identity;
    float matching_feature_weight = 0.0f;
};

class mm_fake_adapter
{
public:
    using state_type = mm_fake_state;
    using reset_context_type = mm_fake_reset_context;

    bool initialize(std::string& message)
    {
        mm_server_fill_fixed_joint_names(identity_);
        identity_.skeleton_signature = std::string(64, '1');
        identity_.build_commit = "test-adapter";
        identity_.joint_contract_sha256 = std::string(64, '2');
        identity_.motion_manifest_sha256 = std::string(64, '3');
        identity_.database_sha256 = std::string(64, '4');
        identity_.terrain_features_sha256 = std::string(64, '5');
        identity_.terrain_support_sha256 = std::string(64, '6');
        identity_.scene_index_sha256 = std::string(64, '7');
        identity_.coordinate_signature = G1_RuntimeCoordinateSignature;

        joint_feasibility_.frame_count = 1;
        joint_feasibility_.raw_safe.resize(1);
        joint_feasibility_.raw_safe(0) = 1U;
        joint_feasibility_.search_safe.resize(1);
        joint_feasibility_.search_safe(0) = 1U;
        joint_feasibility_.raw_safe_count = 1;
        joint_feasibility_.raw_unsafe_count = 0;
        joint_feasibility_.search_safe_count = 1;
        char error[256] = {};
        if (!sonic_joint_feasibility_digest(
                joint_feasibility_.mask_sha256,
                joint_feasibility_.frame_count,
                joint_feasibility_.raw_safe,
                joint_feasibility_.search_safe,
                error,
                static_cast<int>(sizeof(error)))) {
            message = error;
            return false;
        }
        mm_server_set_joint_feasibility_identity(
            identity_.joint_feasibility, joint_feasibility_);
        return true;
    }

    const mm_server_identity& identity() const { return identity_; }
    const mm_server_scene_identity& scene_identity() const
    {
        return scene_identity_;
    }

    const mm_server_scene_identity& prepared_scene_identity(
        const reset_context_type& context) const
    {
        return context.identity;
    }

    bool prepare_reset(
        state_type& state,
        mm_chunk_boundary& boundary,
        reset_context_type& context,
        const mm_chunk_reset_request& request,
        std::string& message)
    {
        if (request.scene_id == "fail-scene") {
            message = "injected fake scene failure";
            return false;
        }
        if (request.route_id == "fail-route") {
            message = "injected fake route failure";
            return false;
        }
        state.frame = 0;
        state.history.assign(1, 0);
        if (request.scene_id == "fail-reset") {
            message = "injected fake controller reset failure";
            return false;
        }

        context.identity.scene_id = request.scene_id;
        context.identity.route_id = request.route_id;
        context.identity.terrain_weight = request.terrain_weight;
        context.identity.coordinate_signature = G1_RuntimeCoordinateSignature;
        context.identity.heightfield_sha256 = std::string(64, '8');
        context.identity.mesh_sha256 = std::string(64, '9');
        context.identity.walkability_sha256 = std::string(64, 'a');
        context.matching_feature_weight = request.terrain_weight;
        mm_chunk_boundary next_boundary;
        if (!observe(next_boundary, state, message)) return false;
        if (request.scene_id == "fail-observe") {
            message = "injected fake observation failure";
            return false;
        }
        if (request.scene_id == "nonfinite-reset") {
            next_boundary.physical_pelvis_position_holden[0] =
                std::numeric_limits<float>::infinity();
        }
        boundary = next_boundary;
        return true;
    }

    void publish_reset(reset_context_type& context)
    {
        using std::swap;
        swap(scene_identity_, context.identity);
        swap(active_matching_feature_weight_, context.matching_feature_weight);
    }

    bool clone(
        state_type& destination,
        const state_type& source,
        std::string&)
    {
        destination = source;
        return true;
    }

    void swap(state_type& first, state_type& second)
    {
        using std::swap;
        swap(first, second);
    }

    bool observe(
        mm_chunk_boundary& boundary,
        const state_type& state,
        std::string&)
    {
        boundary = mm_chunk_boundary();
        for (int joint = 0; joint < MM_CHUNK_JOINT_COUNT; ++joint) {
            boundary.joint_position_source[joint] =
                static_cast<float>(state.frame) * 0.01f +
                static_cast<float>(joint) * 0.001f;
            boundary.joint_velocity_source[joint] =
                static_cast<float>(state.frame) * 0.02f -
                static_cast<float>(joint) * 0.001f;
        }
        boundary.physical_pelvis_position_holden[0] =
            static_cast<float>(state.frame) * 0.01f;
        boundary.physical_pelvis_position_holden[1] = 0.8f;
        boundary.physical_pelvis_orientation_holden[0] = 1.0f;
        boundary.virtual_root_position_holden[2] =
            static_cast<float>(state.frame) * 0.02f;
        boundary.virtual_root_orientation_holden[0] = 1.0f;
        return true;
    }

    bool advance(
        mm_chunk_step_diagnostic& diagnostic,
        state_type& state,
        const mm_chunk_generate_request& request,
        int step,
        std::string& message)
    {
        if (request.candidate_id == "fail-step" && step == 4) {
            message = "injected fake generation failure";
            return false;
        }
        ++state.frame;
        state.history.push_back(state.frame);
        diagnostic = mm_chunk_step_diagnostic();
        diagnostic.selected_database_frame = 1000 + state.frame;
        diagnostic.searched = (step % 3) == 0;
        diagnostic.transitioned = step == 6;
        diagnostic.terrain_cost = active_matching_feature_weight_ +
                                  static_cast<float>(step) * 0.25f;
        if (request.candidate_id == "nonfinite-generate" && step == 6) {
            diagnostic.terrain_cost =
                std::numeric_limits<float>::infinity();
        }
        diagnostic.applied_velocity_holden[0] =
            request.requested_velocity_holden[0];
        diagnostic.applied_velocity_holden[1] =
            request.requested_velocity_holden[1];
        diagnostic.applied_velocity_holden[2] =
            request.requested_velocity_holden[2];
        diagnostic.support_height = static_cast<float>(state.frame) * 0.01f;
        diagnostic.support_target = diagnostic.support_height;
        for (int sample = 0; sample < MM_CHUNK_TERRAIN_SAMPLE_COUNT;
             ++sample) {
            diagnostic.terrain_values[sample] =
                static_cast<float>(state.frame + sample) * 0.01f;
            diagnostic.terrain_points_holden[sample][0] =
                static_cast<float>(sample) * 0.25f;
            diagnostic.terrain_points_holden[sample][1] =
                diagnostic.terrain_values[sample];
            diagnostic.terrain_points_holden[sample][2] =
                static_cast<float>(state.frame) * 0.02f;
        }
        return true;
    }

private:
    mm_server_identity identity_;
    mm_server_scene_identity scene_identity_;
    float active_matching_feature_weight_ = 0.0f;
    sonic_joint_feasibility_certificate joint_feasibility_;
};

static void mm_json_write_string_array(
    mm_chunk_json_writer& writer,
    const std::vector<std::string>& values)
{
    writer.character('[');
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) writer.character(',');
        writer.string(values[index]);
    }
    writer.character(']');
}

static void mm_json_write_float_array(
    mm_chunk_json_writer& writer,
    const float* values,
    int count)
{
    writer.character('[');
    for (int index = 0; index < count; ++index) {
        if (index != 0) writer.character(',');
        writer.number(values[index]);
    }
    writer.character(']');
}

static void mm_json_write_boundary(
    mm_chunk_json_writer& writer,
    const mm_chunk_boundary& boundary,
    const std::vector<std::string>& source_joint_names,
    const std::string& session_id)
{
    writer.raw("{\"session_id\":");
    writer.string(session_id);
    writer.raw(",\"source_joint_names\":");
    mm_json_write_string_array(writer, source_joint_names);
    writer.raw(",\"joint_position_source\":");
    mm_json_write_float_array(
        writer, boundary.joint_position_source, MM_CHUNK_JOINT_COUNT);
    writer.raw(",\"joint_velocity_source\":");
    mm_json_write_float_array(
        writer, boundary.joint_velocity_source, MM_CHUNK_JOINT_COUNT);
    writer.raw(",\"physical_pelvis_position_holden\":");
    mm_json_write_float_array(
        writer, boundary.physical_pelvis_position_holden, 3);
    writer.raw(",\"physical_pelvis_orientation_holden\":");
    mm_json_write_float_array(
        writer, boundary.physical_pelvis_orientation_holden, 4);
    writer.raw(",\"virtual_root_position_holden\":");
    mm_json_write_float_array(writer, boundary.virtual_root_position_holden, 3);
    writer.raw(",\"virtual_root_orientation_holden\":");
    mm_json_write_float_array(
        writer, boundary.virtual_root_orientation_holden, 4);
    writer.character('}');
}

static void mm_json_write_scene(
    mm_chunk_json_writer& writer,
    const mm_server_scene_identity& scene)
{
    writer.raw("{\"scene_id\":");
    writer.string(scene.scene_id);
    writer.raw(",\"route_id\":");
    writer.string(scene.route_id);
    writer.raw(",\"terrain_weight\":");
    writer.number(scene.terrain_weight);
    writer.raw(",\"coordinate_signature\":");
    writer.string(scene.coordinate_signature);
    writer.raw(",\"heightfield_sha256\":");
    writer.string(scene.heightfield_sha256);
    writer.raw(",\"mesh_sha256\":");
    writer.string(scene.mesh_sha256);
    writer.raw(",\"walkability_sha256\":");
    writer.string(scene.walkability_sha256);
    writer.character('}');
}

static void mm_json_write_artifacts(
    mm_chunk_json_writer& writer,
    const mm_server_identity& identity)
{
    writer.raw("{\"build_commit\":");
    writer.string(identity.build_commit);
    writer.raw(",\"joint_contract_sha256\":");
    writer.string(identity.joint_contract_sha256);
    writer.raw(",\"motion_manifest_sha256\":");
    writer.string(identity.motion_manifest_sha256);
    writer.raw(",\"database_sha256\":");
    writer.string(identity.database_sha256);
    writer.raw(",\"terrain_features_sha256\":");
    writer.string(identity.terrain_features_sha256);
    writer.raw(",\"terrain_support_sha256\":");
    writer.string(identity.terrain_support_sha256);
    writer.raw(",\"scene_index_sha256\":");
    writer.string(identity.scene_index_sha256);
    writer.raw(",\"skeleton_signature\":");
    writer.string(identity.skeleton_signature);
    writer.raw(",\"coordinate_signature\":");
    writer.string(identity.coordinate_signature);
    writer.character('}');
}

static void mm_json_write_joint_feasibility(
    mm_chunk_json_writer& writer,
    const mm_server_joint_feasibility_identity& identity)
{
    writer.raw("{\"schema\":");
    writer.string(MM_CHUNK_JOINT_FEASIBILITY_SCHEMA);
    writer.raw(",\"frame_count\":");
    writer.integer(identity.frame_count);
    writer.raw(",\"raw_safe_count\":");
    writer.integer(identity.raw_safe_count);
    writer.raw(",\"raw_unsafe_count\":");
    writer.integer(identity.raw_unsafe_count);
    writer.raw(",\"search_safe_count\":");
    writer.integer(identity.search_safe_count);
    writer.raw(",\"mask_sha256\":");
    writer.string(identity.mask_sha256);
    writer.raw(",\"joint_limit_violation_count\":[");
    for (int joint = 0; joint < SonicG1JointCount; ++joint) {
        if (joint != 0) writer.character(',');
        writer.integer(identity.joint_limit_violation_count[joint]);
    }
    writer.raw("]}");
}

static std::string mm_json_hello_data(const mm_server_identity& identity)
{
    mm_chunk_json_writer writer;
    writer.raw("{\"schema\":");
    writer.string(MM_CHUNK_SCHEMA);
    writer.raw(",\"protocol_version\":1,\"source_joint_names\":");
    mm_json_write_string_array(writer, identity.source_joint_names);
    writer.raw(",\"target_joint_names\":");
    mm_json_write_string_array(writer, identity.target_joint_names);
    writer.raw(",\"skeleton_signature\":");
    writer.string(identity.skeleton_signature);
    writer.raw(",\"source_rate_hz\":25,\"supported_source_intervals\":[10],"
               "\"build_commit\":");
    writer.string(identity.build_commit);
    writer.raw(",\"joint_contract_sha256\":");
    writer.string(identity.joint_contract_sha256);
    writer.raw(",\"motion_manifest_sha256\":");
    writer.string(identity.motion_manifest_sha256);
    writer.raw(",\"database_sha256\":");
    writer.string(identity.database_sha256);
    writer.raw(",\"terrain_features_sha256\":");
    writer.string(identity.terrain_features_sha256);
    writer.raw(",\"terrain_support_sha256\":");
    writer.string(identity.terrain_support_sha256);
    writer.raw(",\"scene_index_sha256\":");
    writer.string(identity.scene_index_sha256);
    writer.raw(",\"coordinate_signature\":");
    writer.string(identity.coordinate_signature);
    writer.raw(",\"joint_feasibility\":");
    mm_json_write_joint_feasibility(writer, identity.joint_feasibility);
    writer.character('}');
    return writer.valid() ? writer.text() : std::string();
}

static std::string mm_json_reset_data(
    const std::string& session_id,
    const mm_chunk_boundary& boundary,
    const mm_server_identity& identity,
    const mm_server_scene_identity& scene)
{
    mm_chunk_json_writer writer;
    writer.raw("{\"session_id\":");
    writer.string(session_id);
    writer.raw(",\"active_candidate_id\":null,\"scene\":");
    mm_json_write_scene(writer, scene);
    writer.raw(",\"initial_boundary\":");
    mm_json_write_boundary(
        writer, boundary, identity.source_joint_names, session_id);
    writer.character('}');
    return writer.valid() ? writer.text() : std::string();
}

enum mm_boundary_field
{
    mm_joint_position,
    mm_joint_velocity,
    mm_physical_position,
    mm_physical_orientation,
    mm_virtual_position,
    mm_virtual_orientation,
};

static void mm_json_write_boundary_matrix(
    mm_chunk_json_writer& writer,
    const std::vector<mm_chunk_boundary>& boundaries,
    mm_boundary_field field)
{
    writer.character('[');
    for (std::size_t index = 0; index < boundaries.size(); ++index) {
        if (index != 0) writer.character(',');
        const mm_chunk_boundary& boundary = boundaries[index];
        switch (field) {
        case mm_joint_position:
            mm_json_write_float_array(
                writer,
                boundary.joint_position_source,
                MM_CHUNK_JOINT_COUNT);
            break;
        case mm_joint_velocity:
            mm_json_write_float_array(
                writer,
                boundary.joint_velocity_source,
                MM_CHUNK_JOINT_COUNT);
            break;
        case mm_physical_position:
            mm_json_write_float_array(
                writer, boundary.physical_pelvis_position_holden, 3);
            break;
        case mm_physical_orientation:
            mm_json_write_float_array(
                writer, boundary.physical_pelvis_orientation_holden, 4);
            break;
        case mm_virtual_position:
            mm_json_write_float_array(
                writer, boundary.virtual_root_position_holden, 3);
            break;
        default:
            mm_json_write_float_array(
                writer, boundary.virtual_root_orientation_holden, 4);
            break;
        }
    }
    writer.character(']');
}

static void mm_json_write_step_scalar_array(
    mm_chunk_json_writer& writer,
    const std::vector<mm_chunk_step_diagnostic>& steps,
    int field)
{
    writer.character('[');
    for (std::size_t index = 0; index < steps.size(); ++index) {
        if (index != 0) writer.character(',');
        const mm_chunk_step_diagnostic& step = steps[index];
        if (field == 0) writer.integer(step.selected_database_frame);
        else if (field == 1) writer.boolean(step.searched);
        else if (field == 2) writer.boolean(step.transitioned);
        else if (field == 3) writer.number(step.terrain_cost);
        else if (field == 4) writer.number(step.support_height);
        else writer.number(step.support_target);
    }
    writer.character(']');
}

static void mm_json_write_terrain_values(
    mm_chunk_json_writer& writer,
    const std::vector<mm_chunk_step_diagnostic>& steps)
{
    writer.character('[');
    for (std::size_t step = 0; step < steps.size(); ++step) {
        if (step != 0) writer.character(',');
        mm_json_write_float_array(
            writer,
            steps[step].terrain_values,
            MM_CHUNK_TERRAIN_SAMPLE_COUNT);
    }
    writer.character(']');
}

static void mm_json_write_terrain_points(
    mm_chunk_json_writer& writer,
    const std::vector<mm_chunk_step_diagnostic>& steps)
{
    writer.character('[');
    for (std::size_t step = 0; step < steps.size(); ++step) {
        if (step != 0) writer.character(',');
        writer.character('[');
        for (int sample = 0; sample < MM_CHUNK_TERRAIN_SAMPLE_COUNT;
             ++sample) {
            if (sample != 0) writer.character(',');
            mm_json_write_float_array(
                writer, steps[step].terrain_points_holden[sample], 3);
        }
        writer.character(']');
    }
    writer.character(']');
}

static void mm_json_write_applied_velocities(
    mm_chunk_json_writer& writer,
    const std::vector<mm_chunk_step_diagnostic>& steps)
{
    writer.character('[');
    for (std::size_t step = 0; step < steps.size(); ++step) {
        if (step != 0) writer.character(',');
        mm_json_write_float_array(
            writer, steps[step].applied_velocity_holden, 3);
    }
    writer.character(']');
}

static std::string mm_json_generate_data(
    const mm_chunk_generate_request& request,
    const mm_chunk_candidate& candidate,
    const mm_server_identity& identity,
    const mm_server_scene_identity& scene)
{
    mm_chunk_json_writer writer;
    writer.raw("{\"schema\":");
    writer.string(MM_CHUNK_SCHEMA);
    writer.raw(",\"session_id\":");
    writer.string(request.session_id);
    writer.raw(",\"candidate_id\":");
    writer.string(request.candidate_id);
    writer.raw(",\"predecessor_id\":");
    if (request.predecessor_is_null) writer.raw("null");
    else writer.string(request.predecessor_id);
    writer.raw(",\"source_rate_hz\":25,\"source_intervals\":10,"
               "\"timestamps_s\":[");
    for (int boundary = 0; boundary <= MM_CHUNK_SOURCE_INTERVALS;
         ++boundary) {
        if (boundary != 0) writer.character(',');
        writer.number(
            static_cast<float>(boundary) /
            static_cast<float>(MM_CHUNK_SOURCE_RATE_HZ));
    }
    writer.raw("],\"source_joint_names\":");
    mm_json_write_string_array(writer, identity.source_joint_names);
    writer.raw(",\"target_joint_names\":");
    mm_json_write_string_array(writer, identity.target_joint_names);
    writer.raw(",\"joint_position_source\":");
    mm_json_write_boundary_matrix(writer, candidate.boundaries, mm_joint_position);
    writer.raw(",\"joint_velocity_source\":");
    mm_json_write_boundary_matrix(writer, candidate.boundaries, mm_joint_velocity);
    writer.raw(",\"physical_pelvis_position_holden\":");
    mm_json_write_boundary_matrix(
        writer, candidate.boundaries, mm_physical_position);
    writer.raw(",\"physical_pelvis_orientation_holden\":");
    mm_json_write_boundary_matrix(
        writer, candidate.boundaries, mm_physical_orientation);
    writer.raw(",\"virtual_root_position_holden\":");
    mm_json_write_boundary_matrix(
        writer, candidate.boundaries, mm_virtual_position);
    writer.raw(",\"virtual_root_orientation_holden\":");
    mm_json_write_boundary_matrix(
        writer, candidate.boundaries, mm_virtual_orientation);
    writer.raw(",\"selected_database_frame\":");
    mm_json_write_step_scalar_array(writer, candidate.steps, 0);
    writer.raw(",\"searched\":");
    mm_json_write_step_scalar_array(writer, candidate.steps, 1);
    writer.raw(",\"transitioned\":");
    mm_json_write_step_scalar_array(writer, candidate.steps, 2);
    writer.raw(",\"terrain_cost\":");
    mm_json_write_step_scalar_array(writer, candidate.steps, 3);
    writer.raw(",\"terrain_values\":");
    mm_json_write_terrain_values(writer, candidate.steps);
    writer.raw(",\"terrain_points_holden\":");
    mm_json_write_terrain_points(writer, candidate.steps);
    writer.raw(",\"support_height\":");
    mm_json_write_step_scalar_array(writer, candidate.steps, 4);
    writer.raw(",\"support_target\":");
    mm_json_write_step_scalar_array(writer, candidate.steps, 5);
    writer.raw(",\"scene\":");
    mm_json_write_scene(writer, scene);
    writer.raw(",\"command\":{\"requested_velocity_holden\":");
    mm_json_write_float_array(writer, request.requested_velocity_holden, 3);
    writer.raw(",\"desired_heading_holden_wxyz\":");
    mm_json_write_float_array(writer, request.desired_heading_holden_wxyz, 4);
    writer.raw(",\"applied_velocity_holden\":");
    mm_json_write_applied_velocities(writer, candidate.steps);
    writer.raw("},\"artifacts\":");
    mm_json_write_artifacts(writer, identity);
    writer.character('}');
    return writer.valid() ? writer.text() : std::string();
}

static std::string mm_json_finish_data(
    const std::string& session_id,
    const std::string& candidate_id,
    const std::string& active_candidate_id)
{
    mm_chunk_json_writer writer;
    writer.raw("{\"session_id\":");
    writer.string(session_id);
    writer.raw(",\"candidate_id\":");
    writer.string(candidate_id);
    writer.raw(",\"active_candidate_id\":");
    if (active_candidate_id.empty()) writer.raw("null");
    else writer.string(active_candidate_id);
    writer.character('}');
    return writer.text();
}

static std::string mm_json_success_response(
    const std::string& operation,
    const std::string& request_id,
    const std::string& data)
{
    mm_chunk_json_writer writer;
    writer.raw("{\"v\":1,\"ok\":true,\"op\":");
    writer.string(operation);
    writer.raw(",\"request_id\":");
    writer.string(request_id);
    writer.raw(",\"data\":");
    writer.raw(data.empty() ? "{}" : data);
    writer.character('}');
    return writer.valid() ? writer.text() : std::string();
}

static std::string mm_json_error_response(
    const std::string& operation,
    const std::string& request_id,
    const mm_chunk_error& error)
{
    mm_chunk_json_writer writer;
    writer.raw("{\"v\":1,\"ok\":false,\"op\":");
    writer.string(operation.empty() ? "<invalid>" : operation);
    writer.raw(",\"request_id\":");
    writer.string(request_id);
    writer.raw(",\"error\":{\"code\":");
    writer.string(error.code.empty() ? "internal_error" : error.code);
    writer.raw(",\"message\":");
    writer.string(error.message.empty() ? "unspecified protocol error" :
                                       error.message);
    writer.raw("}}");
    return writer.text();
}

static bool mm_server_emit(const std::string& response)
{
    return !response.empty() &&
           std::fwrite(response.data(), 1, response.size(), stdout) ==
               response.size() &&
           std::fputc('\n', stdout) != EOF;
}

template<typename Adapter>
static int mm_server_run(Adapter& adapter)
{
    using protocol_type = mm_chunk_protocol<Adapter>;
    protocol_type protocol(adapter);
    std::string line;
    while (std::getline(std::cin, line)) {
        mm_chunk_request request;
        mm_chunk_error error;
        if (!mm_chunk_json_parse_request(request, line, error)) {
            if (!mm_server_emit(mm_json_error_response(
                    request.operation_text, request.request_id, error))) {
                return 2;
            }
            continue;
        }

        bool success = false;
        bool stop = false;
        std::string data;
        if (request.operation == mm_chunk_op_hello) {
            success = protocol.hello(error);
            if (success) data = mm_json_hello_data(adapter.identity());
        } else if (request.operation == mm_chunk_op_reset) {
            typename protocol_type::reset_preparation preparation;
            success = protocol.prepare_reset(
                request.reset, preparation, error);
            if (success) {
                data = mm_json_reset_data(
                    request.reset.session_id,
                    preparation.boundary,
                    adapter.identity(),
                    adapter.prepared_scene_identity(
                        preparation.adapter_context));
                if (data.empty()) {
                    success = false;
                    mm_chunk_fail(
                        error,
                        "serialization_failed",
                        "response contains a non-finite or unserializable value");
                } else {
                    success = protocol.publish_reset(preparation, error);
                }
            }
        } else if (request.operation == mm_chunk_op_generate) {
            typename protocol_type::generate_preparation preparation;
            success = protocol.prepare_generate(
                request.generate, preparation, error);
            if (success) {
                data = mm_json_generate_data(
                    request.generate,
                    preparation.candidate,
                    adapter.identity(),
                    adapter.scene_identity());
                if (data.empty()) {
                    success = false;
                    mm_chunk_fail(
                        error,
                        "serialization_failed",
                        "response contains a non-finite or unserializable value");
                } else {
                    success = protocol.publish_generate(preparation, error);
                }
            }
        } else if (request.operation == mm_chunk_op_commit) {
            success = protocol.commit(
                request.session_id, request.candidate_id, error);
            if (success) {
                data = mm_json_finish_data(
                    request.session_id,
                    request.candidate_id,
                    protocol.session().active_candidate_id);
            }
        } else if (request.operation == mm_chunk_op_abort) {
            success = protocol.abort(
                request.session_id, request.candidate_id, error);
            if (success) {
                data = mm_json_finish_data(
                    request.session_id,
                    request.candidate_id,
                    protocol.session().active_candidate_id);
            }
        } else if (request.operation == mm_chunk_op_close) {
            success = protocol.close(error);
            stop = success;
            if (success) data = "{}";
        }

        if (success && data.empty()) {
            success = false;
            mm_chunk_fail(
                error,
                "serialization_failed",
                "response contains a non-finite or unserializable value");
        }
        const std::string response = success
            ? mm_json_success_response(
                  request.operation_text, request.request_id, data)
            : mm_json_error_response(
                  request.operation_text, request.request_id, error);
        if (!mm_server_emit(response)) return 2;
        if (stop) break;
    }
    if (std::cin.bad()) {
        std::fprintf(stderr, "MM chunk server: stdin read failed\n");
        return 2;
    }
    return 0;
}

int main()
{
    if (std::setlocale(LC_ALL, "C") == nullptr) {
        std::fprintf(stderr, "MM chunk server: cannot set C locale\n");
        return 2;
    }
    if (std::setvbuf(stdout, nullptr, _IONBF, 0) != 0) {
        std::fprintf(stderr, "MM chunk server: cannot disable stdout buffering\n");
        return 2;
    }

    const char* test_adapter = std::getenv("SONIC_MM_TEST_ADAPTER");
    if (test_adapter != nullptr && std::strcmp(test_adapter, "1") == 0) {
        std::fprintf(stderr, "MM chunk server: using deterministic test adapter\n");
        mm_fake_adapter adapter;
        std::string error;
        if (!adapter.initialize(error)) {
            std::fprintf(
                stderr,
                "MM chunk server test adapter error: %s\n",
                error.c_str());
            return 2;
        }
        return mm_server_run(adapter);
    }

    const char* terrain_root = std::getenv("SONIC_TERRAIN_DIR");
    if (terrain_root == nullptr || terrain_root[0] == '\0') {
        std::fprintf(
            stderr,
            "MM chunk server: SONIC_TERRAIN_DIR is required for real artifacts\n");
        return 2;
    }
    const char* contract_path = std::getenv("SONIC_JOINT_CONTRACT");
    if (contract_path == nullptr || contract_path[0] == '\0') {
        contract_path = MM_CHUNK_DEFAULT_JOINT_CONTRACT;
    }
    const char* registry_path = std::getenv("SONIC_SCENE_REGISTRY");
    if (registry_path == nullptr || registry_path[0] == '\0') {
        registry_path = MM_CHUNK_DEFAULT_SCENE_REGISTRY;
    }
    mm_real_adapter adapter;
    std::string error;
    if (!adapter.initialize(
            terrain_root, contract_path, registry_path, error)) {
        std::fprintf(
            stderr,
            "MM chunk server artifact error: %s\n",
            error.c_str());
        return 2;
    }
    std::fprintf(stderr, "MM chunk server: real artifacts loaded\n");
    return mm_server_run(adapter);
}
