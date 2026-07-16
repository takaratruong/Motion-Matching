#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "sonic/cpp/mm_chunk_json.h"
#include "sonic/cpp/g1_joint_projection.h"
#include "sonic/cpp/g1_runtime.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifndef MM_CHUNK_BUILD_COMMIT
#define MM_CHUNK_BUILD_COMMIT "unknown"
#endif

#ifndef MM_CHUNK_DEFAULT_JOINT_CONTRACT
#define MM_CHUNK_DEFAULT_JOINT_CONTRACT "sonic/configs/g1_joint_contract.json"
#endif

static constexpr const char* MM_CHUNK_SCHEMA = "mm-chunk/v1";

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

template<typename T>
static bool mm_server_animation_shape_is_valid(
    const array2d<T>& values,
    int frames,
    int bones,
    const char* label,
    char* error,
    int capacity)
{
    if (values.rows != frames || values.cols != bones ||
        values.data == nullptr) {
        return scene_error(
            error,
            capacity,
            "G1 database %s shape mismatch: expected %dx%d, got %dx%d",
            label,
            frames,
            bones,
            values.rows,
            values.cols);
    }
    return true;
}

static bool mm_server_database_is_valid(
    const database& db,
    char* error,
    int capacity)
{
    const int frames = db.nframes();
    const int bones = db.nbones();
    if (frames <= 0) {
        return scene_error(error, capacity, "G1 database has no frames");
    }
    if (!mm_server_animation_shape_is_valid(
            db.bone_positions,
            frames,
            bones,
            "bone_positions",
            error,
            capacity) ||
        !mm_server_animation_shape_is_valid(
            db.bone_velocities,
            frames,
            bones,
            "bone_velocities",
            error,
            capacity) ||
        !mm_server_animation_shape_is_valid(
            db.bone_rotations,
            frames,
            bones,
            "bone_rotations",
            error,
            capacity) ||
        !mm_server_animation_shape_is_valid(
            db.bone_angular_velocities,
            frames,
            bones,
            "bone_angular_velocities",
            error,
            capacity)) {
        return false;
    }
    if (db.contact_states.rows != frames || db.contact_states.cols != 2 ||
        db.contact_states.data == nullptr) {
        return scene_error(
            error,
            capacity,
            "G1 database contact shape mismatch: expected %dx2, got %dx%d",
            frames,
            db.contact_states.rows,
            db.contact_states.cols);
    }
    if (db.range_starts.size <= 0 ||
        db.range_stops.size != db.range_starts.size ||
        db.range_starts.data == nullptr || db.range_stops.data == nullptr) {
        return scene_error(
            error,
            capacity,
            "G1 database range arrays must be nonempty and equal-sized");
    }
    int expected_start = 0;
    for (int range = 0; range < db.nranges(); ++range) {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        if (start != expected_start || stop <= start || stop > frames) {
            return scene_error(
                error,
                capacity,
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
    if (expected_start != frames) {
        return scene_error(
            error,
            capacity,
            "G1 database ranges stop at %d instead of covering %d frames",
            expected_start,
            frames);
    }
    return true;
}

static bool mm_server_feature_value_is_safe(float value)
{
    const std::uint32_t magnitude =
        feature_float_bits(value) & UINT32_C(0x7fffffff);
    return feature_float_is_finite(value) &&
           magnitude != UINT32_C(0x7f7fffff);
}

static bool mm_server_matching_features_are_valid(
    const database& db,
    char* error,
    int capacity)
{
    static constexpr int expected_features = 31;
    if (db.features.rows != db.nframes() ||
        db.features.cols != expected_features || db.features.data == nullptr ||
        db.features_offset.size != expected_features ||
        db.features_scale.size != expected_features ||
        db.features_offset.data == nullptr ||
        db.features_scale.data == nullptr) {
        return scene_error(
            error,
            capacity,
            "G1 matching feature build failed: expected %dx%d features, "
            "got %dx%d",
            db.nframes(),
            expected_features,
            db.features.rows,
            db.features.cols);
    }
    for (int feature = 0; feature < expected_features; ++feature) {
        if (!mm_server_feature_value_is_safe(db.features_offset(feature)) ||
            !feature_float_is_positive_finite(db.features_scale(feature))) {
            return scene_error(
                error,
                capacity,
                "G1 matching feature %d has invalid offset/scale",
                feature);
        }
    }
    for (int index = 0; index < db.features.rows * db.features.cols;
         ++index) {
        if (!mm_server_feature_value_is_safe(db.features.data[index])) {
            return scene_error(
                error,
                capacity,
                "G1 matching feature row payload is invalid at value %d",
                index);
        }
    }

    const int small_rows =
        (db.nframes() + BOUND_SM_SIZE - 1) / BOUND_SM_SIZE;
    const int large_rows =
        (db.nframes() + BOUND_LR_SIZE - 1) / BOUND_LR_SIZE;
    const array2d<float>* bounds[4] = {
        &db.bound_sm_min,
        &db.bound_sm_max,
        &db.bound_lr_min,
        &db.bound_lr_max,
    };
    const int expected_rows[4] = {
        small_rows, small_rows, large_rows, large_rows};
    for (int bound = 0; bound < 4; ++bound) {
        if (bounds[bound]->rows != expected_rows[bound] ||
            bounds[bound]->cols != expected_features ||
            bounds[bound]->data == nullptr) {
            return scene_error(
                error,
                capacity,
                "G1 matching bound %d shape mismatch: expected %dx%d, "
                "got %dx%d",
                bound,
                expected_rows[bound],
                expected_features,
                bounds[bound]->rows,
                bounds[bound]->cols);
        }
        for (int index = 0;
             index < bounds[bound]->rows * bounds[bound]->cols;
             ++index) {
            if (!mm_server_feature_value_is_safe(
                    bounds[bound]->data[index])) {
                return scene_error(
                    error,
                    capacity,
                    "G1 matching bound %d has invalid value at %d",
                    bound,
                    index);
            }
        }
    }
    for (int index = 0; index < small_rows * expected_features; ++index) {
        if (db.bound_sm_min.data[index] > db.bound_sm_max.data[index]) {
            return scene_error(
                error, capacity, "G1 small matching bounds are inverted");
        }
    }
    for (int index = 0; index < large_rows * expected_features; ++index) {
        if (db.bound_lr_min.data[index] > db.bound_lr_max.data[index]) {
            return scene_error(
                error, capacity, "G1 large matching bounds are inverted");
        }
    }
    return true;
}

static int mm_server_bone_index(const std::string& name)
{
    static const char* const names[G1_BoneCount] = {
        "Simulation", "Hips", "LeftHipPitch", "LeftHipRoll",
        "LeftHipYaw", "LeftKnee", "LeftAnkle", "LeftToe",
        "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee",
        "RightAnkle", "RightToe", "Spine", "Spine1", "Spine2",
        "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw",
        "LeftElbow", "LeftWristRoll", "LeftWristPitch", "LeftWrist",
        "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw",
        "RightElbow", "RightWristRoll", "RightWristPitch", "RightWrist",
    };
    for (int index = 0; index < G1_BoneCount; ++index) {
        if (name == names[index]) return index;
    }
    return -1;
}

static bool mm_server_contract_float(
    float& output,
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    return value != nullptr &&
           scene_number_float(output, *value, key, error, capacity);
}

template<std::size_t Size>
static bool mm_server_contract_float_array(
    float (&output)[Size],
    const json_value& object,
    const char* key,
    char* error,
    int capacity)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr || value->kind != json_array ||
        value->array_value.size() != Size) {
        return scene_error(
            error, capacity, "joint contract field %s has invalid shape", key);
    }
    for (std::size_t index = 0; index < Size; ++index) {
        if (!scene_number_float(
                output[index], value->array_value[index], key, error, capacity)) {
            return false;
        }
    }
    return true;
}

static bool mm_server_load_joint_contract(
    sonic_joint_contract_entry (&contract)[SonicG1JointCount],
    mm_server_identity& identity,
    const char* path,
    char* error,
    int capacity)
{
    json_value document;
    if (!json_document_load(document, path, error, capacity) ||
        !scene_exact_keys(
            document,
            {"rows", "source_mjcf_sha256", "target_order_source_sha256"},
            "joint contract",
            error,
            capacity)) {
        return false;
    }
    const json_value* source_hash = json_member(document, "source_mjcf_sha256");
    const json_value* target_hash =
        json_member(document, "target_order_source_sha256");
    const json_value* rows = json_member(document, "rows");
    if (source_hash == nullptr || target_hash == nullptr ||
        source_hash->kind != json_string || target_hash->kind != json_string ||
        !scene_sha_is_valid(source_hash->string_value) ||
        !scene_sha_is_valid(target_hash->string_value) || rows == nullptr ||
        rows->kind != json_array ||
        rows->array_value.size() != SonicG1JointCount) {
        return scene_error(
            error, capacity, "joint contract identity or row count is invalid");
    }

    std::vector<std::string> source_names(
        static_cast<std::size_t>(SonicG1JointCount));
    std::vector<std::string> target_names(
        static_cast<std::size_t>(SonicG1JointCount));
    for (int row = 0; row < SonicG1JointCount; ++row) {
        const json_value& encoded =
            rows->array_value[static_cast<std::size_t>(row)];
        if (!scene_exact_keys(
                encoded,
                {"axis_holden", "lower", "qpos_address", "sign",
                 "source_bone", "source_index", "source_joint",
                 "source_parent", "static_local_holden_wxyz",
                 "target_index", "target_name", "upper", "zero_offset"},
                "joint contract row",
                error,
                capacity)) {
            return false;
        }
        sonic_joint_contract_entry entry;
        std::string source_bone;
        std::string source_parent;
        int qpos_address = -1;
        float axis[3] = {};
        float rotation[4] = {};
        if (!scene_member_int(
                entry.source_index,
                encoded,
                "source_index",
                "joint contract row",
                error,
                capacity) ||
            !scene_member_int(
                entry.target_index,
                encoded,
                "target_index",
                "joint contract row",
                error,
                capacity) ||
            !scene_member_int(
                qpos_address,
                encoded,
                "qpos_address",
                "joint contract row",
                error,
                capacity) ||
            qpos_address < 0 ||
            !scene_member_string(
                source_bone,
                encoded,
                "source_bone",
                "joint contract row",
                error,
                capacity) ||
            !scene_member_string(
                source_parent,
                encoded,
                "source_parent",
                "joint contract row",
                error,
                capacity) ||
            !scene_member_string(
                entry.source_joint,
                encoded,
                "source_joint",
                "joint contract row",
                error,
                capacity) ||
            !scene_member_string(
                entry.target_joint,
                encoded,
                "target_name",
                "joint contract row",
                error,
                capacity) ||
            !mm_server_contract_float_array(
                axis, encoded, "axis_holden", error, capacity) ||
            !mm_server_contract_float_array(
                rotation,
                encoded,
                "static_local_holden_wxyz",
                error,
                capacity) ||
            !mm_server_contract_float(
                entry.sign, encoded, "sign", error, capacity) ||
            !mm_server_contract_float(
                entry.zero_offset, encoded, "zero_offset", error, capacity) ||
            !mm_server_contract_float(
                entry.lower, encoded, "lower", error, capacity) ||
            !mm_server_contract_float(
                entry.upper, encoded, "upper", error, capacity)) {
            return false;
        }
        entry.source_bone = mm_server_bone_index(source_bone);
        entry.source_parent = mm_server_bone_index(source_parent);
        entry.axis_holden = vec3(axis[0], axis[1], axis[2]);
        entry.static_local_holden =
            quat(rotation[0], rotation[1], rotation[2], rotation[3]);
        if (entry.source_index < 0 || entry.source_index >= SonicG1JointCount ||
            entry.target_index < 0 || entry.target_index >= SonicG1JointCount ||
            entry.source_bone < 0 || entry.source_parent < 0 ||
            !source_names[static_cast<std::size_t>(entry.source_index)].empty() ||
            !target_names[static_cast<std::size_t>(entry.target_index)].empty()) {
            return scene_error(
                error, capacity, "joint contract row %d identity is invalid", row);
        }
        source_names[static_cast<std::size_t>(entry.source_index)] =
            entry.source_joint;
        target_names[static_cast<std::size_t>(entry.target_index)] =
            entry.target_joint;
        contract[row] = entry;
    }
    if (!sonic_projection_contract_valid(contract, error, capacity)) {
        return false;
    }
    mm_server_identity fixed;
    mm_server_fill_fixed_joint_names(fixed);
    if (source_names != fixed.source_joint_names ||
        target_names != fixed.target_joint_names) {
        return scene_error(
            error, capacity, "joint contract source/target order is not fixed");
    }
    identity.source_joint_names = source_names;
    identity.target_joint_names = target_names;
    if (!sha256_file_hex(identity.joint_contract_sha256, path, error, capacity)) {
        return false;
    }
    return true;
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

class mm_real_adapter
{
public:
    using state_type = g1_controller_state;

    bool initialize(
        const char* terrain_root,
        const char* contract_path,
        std::string& message)
    {
        char error[1024] = {};
        terrain_root_ = terrain_root != nullptr ? terrain_root : "";
        identity_.build_commit = MM_CHUNK_BUILD_COMMIT;
        identity_.skeleton_signature = G1_SkeletonSignature;
        if (terrain_root_.empty() || contract_path == nullptr ||
            contract_path[0] == '\0') {
            message = "terrain root and joint contract are required";
            return false;
        }
        std::string manifest_path;
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
            !mm_server_load_joint_contract(
                contract_,
                identity_,
                contract_path,
                error,
                static_cast<int>(sizeof(error)))) {
            message = error;
            return false;
        }
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
        if (!mm_server_database_is_valid(
                database_, error, static_cast<int>(sizeof(error))) ||
            terrain_rows.values.rows != database_.nframes() ||
            terrain_rows.values.cols != 4) {
            message = error[0] != '\0'
                ? error
                : "database and terrain feature shapes differ";
            return false;
        }
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

    bool reset(
        state_type& output,
        mm_chunk_boundary& boundary,
        const mm_chunk_reset_request& request,
        std::string& message)
    {
        char error[1024] = {};
        const int scene_index =
            scene_catalog_find(catalog_, request.scene_id.c_str());
        if (scene_index < 0) {
            message = "unknown scene: " + request.scene_id;
            return false;
        }
        scene_pack next_scene;
        if (!scene_pack_load(
                next_scene,
                terrain_root_.c_str(),
                manifest_,
                catalog_,
                scene_index,
                error,
                static_cast<int>(sizeof(error)))) {
            message = error;
            return false;
        }
        const scene_route* route =
            scene_route_find(next_scene.metadata, request.route_id.c_str());
        if (route == nullptr) {
            message = "scene " + request.scene_id + " has no route " +
                      request.route_id;
            return false;
        }

        state_type next_state;
        if (!g1_controller_state_reset(
                next_state,
                database_,
                support_,
                next_scene,
                error,
                static_cast<int>(sizeof(error)))) {
            message = error;
            return false;
        }
        next_state.route_index = static_cast<int>(
            route - next_scene.metadata.routes.data());
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
        if (!observe_boundary(boundary, next_state, error, sizeof(error)) ||
            !rebuild_matching_features(
                request.terrain_weight, error, sizeof(error))) {
            message = error;
            return false;
        }

        scene_pack_swap(active_scene_, next_scene);
        g1_controller_state_swap(output, next_state);
        scene_identity_.scene_id = request.scene_id;
        scene_identity_.route_id = request.route_id;
        scene_identity_.terrain_weight = request.terrain_weight;
        scene_identity_.coordinate_signature =
            active_scene_.metadata.coordinate_signature;
        scene_identity_.heightfield_sha256 =
            active_scene_.metadata.heightfield.sha256;
        scene_identity_.mesh_sha256 = active_scene_.metadata.mesh.sha256;
        scene_identity_.walkability_sha256 =
            active_scene_.metadata.walkability.sha256;
        return true;
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
        char error[1024] = {};
        if (!g1_runtime_step(
                result,
                state,
                database_,
                support_,
                active_scene_,
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
        char* error,
        int capacity)
    {
        mm_matching_feature_storage prior;
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
        if (!mm_server_matching_features_are_valid(
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
    mm_server_identity identity_;
    mm_server_scene_identity scene_identity_;
    motion_pack_manifest manifest_;
    scene_catalog catalog_;
    scene_pack active_scene_;
    database database_;
    terrain_support_set support_;
    sonic_joint_contract_entry contract_[SonicG1JointCount];
};

struct mm_fake_state
{
    int frame = 0;
    std::vector<int> history;
};

class mm_fake_adapter
{
public:
    using state_type = mm_fake_state;

    mm_fake_adapter()
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
    }

    const mm_server_identity& identity() const { return identity_; }
    const mm_server_scene_identity& scene_identity() const
    {
        return scene_identity_;
    }

    bool reset(
        state_type& state,
        mm_chunk_boundary& boundary,
        const mm_chunk_reset_request& request,
        std::string& message)
    {
        if (request.scene_id == "fail-scene") {
            message = "injected fake reset failure";
            return false;
        }
        state.frame = 0;
        state.history.assign(1, 0);
        scene_identity_.scene_id = request.scene_id;
        scene_identity_.route_id = request.route_id;
        scene_identity_.terrain_weight = request.terrain_weight;
        scene_identity_.coordinate_signature = G1_RuntimeCoordinateSignature;
        scene_identity_.heightfield_sha256 = std::string(64, '8');
        scene_identity_.mesh_sha256 = std::string(64, '9');
        scene_identity_.walkability_sha256 = std::string(64, 'a');
        return observe(boundary, state, message);
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
        diagnostic.terrain_cost = static_cast<float>(step) * 0.25f;
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
    mm_chunk_protocol<Adapter> protocol(adapter);
    std::string line;
    while (std::getline(std::cin, line)) {
        mm_chunk_request request;
        mm_chunk_error error;
        if (line.size() > 1024u * 1024u) {
            request.operation_text = "<invalid>";
            mm_chunk_fail(
                error, "invalid_request", "request exceeds 1 MiB");
            if (!mm_server_emit(mm_json_error_response(
                    request.operation_text, request.request_id, error))) {
                return 2;
            }
            continue;
        }
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
            mm_chunk_boundary initial;
            success = protocol.reset(request.reset, initial, error);
            if (success) {
                data = mm_json_reset_data(
                    request.reset.session_id,
                    initial,
                    adapter.identity(),
                    adapter.scene_identity());
            }
        } else if (request.operation == mm_chunk_op_generate) {
            mm_chunk_candidate candidate;
            success = protocol.generate(request.generate, candidate, error);
            if (success) {
                data = mm_json_generate_data(
                    request.generate,
                    candidate,
                    adapter.identity(),
                    adapter.scene_identity());
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
    mm_real_adapter adapter;
    std::string error;
    if (!adapter.initialize(terrain_root, contract_path, error)) {
        std::fprintf(
            stderr,
            "MM chunk server artifact error: %s\n",
            error.c_str());
        return 2;
    }
    std::fprintf(stderr, "MM chunk server: real artifacts loaded\n");
    return mm_server_run(adapter);
}
