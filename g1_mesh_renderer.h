#pragma once

#include <cstdlib>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "array.h"
#include "g1_kinematic_contract.h"
#include "quat.h"
#include "raylib.h"

enum { G1MeshBoneCount = 39 };

struct G1MeshBinding
{
    int model_to_g1[G1MeshBoneCount];
    Transform bind_local[G1MeshBoneCount];
    int parents[G1MeshBoneCount];
};

struct G1MeshRenderer
{
    Model model;
    ModelAnimation animation;
    G1MeshBinding binding;
    bool loaded;
};

inline constexpr const char* G1MeshG1BoneNames[G1_BoneCount] = {
    nullptr,
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
};

inline constexpr int G1MeshG1Parents[G1_BoneCount] = {
    -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29,
};

static inline bool g1_mesh_error(
    char* error,
    int error_capacity,
    const char* format,
    ...)
{
    if (error != nullptr && error_capacity > 0) {
        std::va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(
            error,
            static_cast<std::size_t>(error_capacity),
            format,
            arguments);
        va_end(arguments);
    }
    return false;
}

static inline void g1_mesh_clear_error(char* error, int error_capacity)
{
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
}

static inline bool g1_mesh_float_is_finite(float value)
{
    static_assert(sizeof(float) == sizeof(std::uint32_t),
                  "G1 mesh renderer requires binary32 floats");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static inline bool g1_mesh_vec3_is_finite(vec3 value)
{
    return g1_mesh_float_is_finite(value.x) &&
           g1_mesh_float_is_finite(value.y) &&
           g1_mesh_float_is_finite(value.z);
}

static inline bool g1_mesh_quat_is_unit(quat value)
{
    if (!g1_mesh_float_is_finite(value.w) ||
        !g1_mesh_float_is_finite(value.x) ||
        !g1_mesh_float_is_finite(value.y) ||
        !g1_mesh_float_is_finite(value.z)) {
        return false;
    }
    const float norm_squared =
        value.w * value.w + value.x * value.x +
        value.y * value.y + value.z * value.z;
    if (!g1_mesh_float_is_finite(norm_squared)) return false;
    const float difference = norm_squared >= 1.0f
        ? norm_squared - 1.0f
        : 1.0f - norm_squared;
    return difference <= 1.0e-4f;
}

static inline vec3 g1_mesh_from_raylib(Vector3 value)
{
    return vec3(value.x, value.y, value.z);
}

static inline Vector3 g1_mesh_to_raylib(vec3 value)
{
    return Vector3{value.x, value.y, value.z};
}

static inline quat g1_mesh_from_raylib(Quaternion value)
{
    return quat(value.w, value.x, value.y, value.z);
}

static inline Quaternion g1_mesh_to_raylib(quat value)
{
    return Quaternion{value.x, value.y, value.z, value.w};
}

static inline Transform g1_mesh_identity_transform()
{
    Transform value = {};
    value.rotation = Quaternion{0.0f, 0.0f, 0.0f, 1.0f};
    value.scale = Vector3{1.0f, 1.0f, 1.0f};
    return value;
}

static inline bool g1_mesh_ranges_overlap(
    const void* first,
    std::size_t first_size,
    const void* second,
    std::size_t second_size)
{
    const std::uintptr_t first_begin =
        reinterpret_cast<std::uintptr_t>(first);
    const std::uintptr_t second_begin =
        reinterpret_cast<std::uintptr_t>(second);
    const std::uintptr_t first_end = first_begin + first_size;
    const std::uintptr_t second_end = second_begin + second_size;
    if (first_end < first_begin || second_end < second_begin) return true;
    return first_begin < second_end && second_begin < first_end;
}

inline int g1_mesh_g1_bone_for_name(const char* name)
{
    if (name == nullptr) return -1;
    for (int g1_bone = G1_Hips;
         g1_bone <= G1_RightWrist;
         ++g1_bone) {
        if (std::strcmp(name, G1MeshG1BoneNames[g1_bone]) == 0) {
            return g1_bone;
        }
    }
    return -1;
}

inline bool g1_mesh_binding_build(
    G1MeshBinding& output,
    const BoneInfo* bones,
    const Transform* global_bind_pose,
    int bone_count,
    char* error,
    int error_capacity)
{
    if (bone_count != G1MeshBoneCount) {
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh requires exactly %d bones, got %d",
            G1MeshBoneCount,
            bone_count);
    }
    if (bones == nullptr || global_bind_pose == nullptr) {
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh bone and bind-pose arrays are required");
    }

    G1MeshBinding candidate = {};
    int mapped_counts[G1_BoneCount] = {};
    for (int model_bone = 0;
         model_bone < G1MeshBoneCount;
         ++model_bone) {
        candidate.model_to_g1[model_bone] =
            g1_mesh_g1_bone_for_name(bones[model_bone].name);
        candidate.parents[model_bone] = bones[model_bone].parent;
        candidate.bind_local[model_bone] = g1_mesh_identity_transform();
        if (candidate.model_to_g1[model_bone] >= 0) {
            ++mapped_counts[candidate.model_to_g1[model_bone]];
        }
        const vec3 bind_translation =
            g1_mesh_from_raylib(global_bind_pose[model_bone].translation);
        const quat bind_rotation =
            g1_mesh_from_raylib(global_bind_pose[model_bone].rotation);
        if (!g1_mesh_vec3_is_finite(bind_translation) ||
            !g1_mesh_quat_is_unit(bind_rotation)) {
            return g1_mesh_error(
                error,
                error_capacity,
                "G1 mesh bind transform %d is invalid",
                model_bone);
        }
    }
    for (int g1_bone = G1_Hips;
         g1_bone <= G1_RightWrist;
         ++g1_bone) {
        if (mapped_counts[g1_bone] != 1) {
            return g1_mesh_error(
                error,
                error_capacity,
                "G1 mesh bone '%s' must occur exactly once",
                G1MeshG1BoneNames[g1_bone]);
        }
    }

    for (int model_bone = 0;
         model_bone < G1MeshBoneCount;
         ++model_bone) {
        const int g1_bone = candidate.model_to_g1[model_bone];
        const int model_parent = candidate.parents[model_bone];
        if (g1_bone == G1_Hips) {
            if (model_parent != -1) {
                return g1_mesh_error(
                    error,
                    error_capacity,
                    "G1 mesh pelvis must be the root bone");
            }
            continue;
        }
        if (g1_bone >= 0) {
            if (model_parent < 0 || model_parent >= G1MeshBoneCount ||
                candidate.model_to_g1[model_parent] !=
                    G1MeshG1Parents[g1_bone]) {
                return g1_mesh_error(
                    error,
                    error_capacity,
                    "G1 mesh mapped parent mismatch for '%s'",
                    bones[model_bone].name);
            }
            continue;
        }
        if (model_parent < 0 || model_parent >= model_bone) {
            return g1_mesh_error(
                error,
                error_capacity,
                "G1 fixed attachment '%s' requires an earlier parent",
                bones[model_bone].name);
        }
        const Transform& parent = global_bind_pose[model_parent];
        const Transform& child = global_bind_pose[model_bone];
        const quat inverse_parent =
            quat_inv(g1_mesh_from_raylib(parent.rotation));
        Transform local = g1_mesh_identity_transform();
        local.rotation = g1_mesh_to_raylib(quat_mul(
            inverse_parent,
            g1_mesh_from_raylib(child.rotation)));
        local.translation = g1_mesh_to_raylib(quat_mul_vec3(
            inverse_parent,
            g1_mesh_from_raylib(child.translation) -
                g1_mesh_from_raylib(parent.translation)));
        candidate.bind_local[model_bone] = local;
    }

    output = candidate;
    g1_mesh_clear_error(error, error_capacity);
    return true;
}

inline bool g1_mesh_pose_build(
    Transform* output,
    int output_count,
    const G1MeshBinding& binding,
    const slice1d<vec3> accepted_positions,
    const slice1d<quat> accepted_rotations,
    char* error,
    int error_capacity)
{
    if (output == nullptr || output_count != G1MeshBoneCount) {
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh pose requires exactly %d output transforms",
            G1MeshBoneCount);
    }
    if (accepted_positions.data == nullptr ||
        accepted_positions.size != G1_BoneCount ||
        accepted_rotations.data == nullptr ||
        accepted_rotations.size != G1_BoneCount) {
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh pose requires exactly %d accepted transforms",
            G1_BoneCount);
    }
    if (g1_mesh_ranges_overlap(
            output,
            sizeof(Transform) * G1MeshBoneCount,
            accepted_positions.data,
            sizeof(vec3) * G1_BoneCount) ||
        g1_mesh_ranges_overlap(
            output,
            sizeof(Transform) * G1MeshBoneCount,
            accepted_rotations.data,
            sizeof(quat) * G1_BoneCount)) {
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh output must not alias accepted pose storage");
    }
    for (int g1_bone = 0; g1_bone < G1_BoneCount; ++g1_bone) {
        if (!g1_mesh_vec3_is_finite(accepted_positions(g1_bone))) {
            return g1_mesh_error(
                error,
                error_capacity,
                "G1 accepted position %d is not finite",
                g1_bone);
        }
        if (!g1_mesh_quat_is_unit(accepted_rotations(g1_bone))) {
            return g1_mesh_error(
                error,
                error_capacity,
                "G1 accepted rotation %d is not unit finite",
                g1_bone);
        }
    }

    Transform candidate[G1MeshBoneCount] = {};
    for (int model_bone = 0;
         model_bone < G1MeshBoneCount;
         ++model_bone) {
        const int g1_bone = binding.model_to_g1[model_bone];
        if (g1_bone >= G1_Hips && g1_bone <= G1_RightWrist) {
            candidate[model_bone].translation =
                g1_mesh_to_raylib(accepted_positions(g1_bone));
            candidate[model_bone].rotation =
                g1_mesh_to_raylib(accepted_rotations(g1_bone));
        } else {
            const int parent = binding.parents[model_bone];
            if (parent < 0 || parent >= model_bone) {
                return g1_mesh_error(
                    error,
                    error_capacity,
                    "G1 fixed attachment %d has an invalid parent",
                    model_bone);
            }
            const quat parent_rotation =
                g1_mesh_from_raylib(candidate[parent].rotation);
            candidate[model_bone].translation = g1_mesh_to_raylib(
                g1_mesh_from_raylib(candidate[parent].translation) +
                quat_mul_vec3(
                    parent_rotation,
                    g1_mesh_from_raylib(
                        binding.bind_local[model_bone].translation)));
            candidate[model_bone].rotation = g1_mesh_to_raylib(quat_mul(
                parent_rotation,
                g1_mesh_from_raylib(
                    binding.bind_local[model_bone].rotation)));
        }
        candidate[model_bone].scale = Vector3{1.0f, 1.0f, 1.0f};
    }

    std::memcpy(
        output,
        candidate,
        sizeof(Transform) * G1MeshBoneCount);
    g1_mesh_clear_error(error, error_capacity);
    return true;
}

inline void g1_mesh_renderer_unload(G1MeshRenderer& renderer)
{
    if (renderer.loaded) {
        ::UnloadModelAnimation(renderer.animation);
        ::UnloadModel(renderer.model);
    }
    renderer = G1MeshRenderer{};
}

static inline void g1_mesh_renderer_discard_partial(
    G1MeshRenderer& renderer)
{
    if (renderer.animation.bones != nullptr ||
        renderer.animation.framePoses != nullptr) {
        ::UnloadModelAnimation(renderer.animation);
    }
    ::UnloadModel(renderer.model);
    renderer = G1MeshRenderer{};
}

inline bool g1_mesh_renderer_load(
    G1MeshRenderer& renderer,
    const char* path,
    char* error,
    int error_capacity)
{
    g1_mesh_renderer_unload(renderer);
    if (path == nullptr || path[0] == '\0') {
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh path may not be empty");
    }

    G1MeshRenderer candidate = {};
    candidate.model = ::LoadModel(path);
    if (candidate.model.meshCount <= 0 || candidate.model.meshes == nullptr ||
        candidate.model.boneCount != G1MeshBoneCount ||
        candidate.model.bones == nullptr ||
        candidate.model.bindPose == nullptr) {
        g1_mesh_renderer_discard_partial(candidate);
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh model layout is unsupported");
    }
    if (!g1_mesh_binding_build(
            candidate.binding,
            candidate.model.bones,
            candidate.model.bindPose,
            candidate.model.boneCount,
            error,
            error_capacity)) {
        g1_mesh_renderer_discard_partial(candidate);
        return false;
    }

    candidate.animation.boneCount = G1MeshBoneCount;
    candidate.animation.bones = static_cast<BoneInfo*>(::MemAlloc(
        static_cast<unsigned int>(
            sizeof(BoneInfo) * G1MeshBoneCount)));
    if (candidate.animation.bones == nullptr) {
        g1_mesh_renderer_discard_partial(candidate);
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh animation bone allocation failed");
    }
    candidate.animation.framePoses = static_cast<Transform**>(::MemAlloc(
        static_cast<unsigned int>(sizeof(Transform*))));
    if (candidate.animation.framePoses == nullptr) {
        g1_mesh_renderer_discard_partial(candidate);
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh animation frame allocation failed");
    }
    candidate.animation.framePoses[0] = nullptr;
    candidate.animation.frameCount = 1;
    candidate.animation.framePoses[0] = static_cast<Transform*>(::MemAlloc(
        static_cast<unsigned int>(
            sizeof(Transform) * G1MeshBoneCount)));
    if (candidate.animation.framePoses[0] == nullptr) {
        g1_mesh_renderer_discard_partial(candidate);
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh animation pose allocation failed");
    }
    std::memcpy(
        candidate.animation.bones,
        candidate.model.bones,
        sizeof(BoneInfo) * G1MeshBoneCount);
    std::memcpy(
        candidate.animation.framePoses[0],
        candidate.model.bindPose,
        sizeof(Transform) * G1MeshBoneCount);
    if (!::IsModelAnimationValid(candidate.model, candidate.animation)) {
        g1_mesh_renderer_discard_partial(candidate);
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh animation hierarchy is invalid");
    }

    candidate.loaded = true;
    renderer = candidate;
    g1_mesh_clear_error(error, error_capacity);
    return true;
}

inline bool g1_mesh_renderer_update(
    G1MeshRenderer& renderer,
    const slice1d<vec3> accepted_positions,
    const slice1d<quat> accepted_rotations,
    char* error,
    int error_capacity)
{
    if (!renderer.loaded || renderer.animation.frameCount != 1 ||
        renderer.animation.framePoses == nullptr ||
        renderer.animation.framePoses[0] == nullptr) {
        return g1_mesh_error(
            error,
            error_capacity,
            "G1 mesh renderer is not loaded");
    }
    if (!g1_mesh_pose_build(
            renderer.animation.framePoses[0],
            renderer.animation.boneCount,
            renderer.binding,
            accepted_positions,
            accepted_rotations,
            error,
            error_capacity)) {
        return false;
    }
    ::UpdateModelAnimation(renderer.model, renderer.animation, 0);
    g1_mesh_clear_error(error, error_capacity);
    return true;
}

inline void g1_mesh_renderer_draw(const G1MeshRenderer& renderer)
{
    if (!renderer.loaded) return;
    ::DrawModel(
        renderer.model,
        Vector3{0.0f, 0.0f, 0.0f},
        1.0f,
        WHITE);
}
