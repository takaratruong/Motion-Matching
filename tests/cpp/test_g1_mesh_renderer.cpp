#include "g1_mesh_renderer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "G1 mesh renderer test failed: %s\n", message);
        std::exit(1);
    }
}

static const char* const fixture_names[G1MeshBoneCount] = {
    "pelvis",
    "imu_in_pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "mid360_link",
    "d435_link",
    "imu_in_torso",
    "head_link",
    "logo_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "pelvis_contour_link",
};

static const int fixture_parents[G1MeshBoneCount] = {
    -1, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    4, 13, 14, 15, 16, 17, 18, 19, 4, 4, 4, 4, 4,
    0, 26, 27, 28, 29, 30, 0, 32, 33, 34, 35, 36, 0,
};

static Quaternion to_raylib_quaternion(quat value)
{
    return Quaternion{value.x, value.y, value.z, value.w};
}

static quat from_raylib_quaternion(Quaternion value)
{
    return quat(value.w, value.x, value.y, value.z);
}

static vec3 from_raylib_vector(Vector3 value)
{
    return vec3(value.x, value.y, value.z);
}

static Vector3 to_raylib_vector(vec3 value)
{
    return Vector3{value.x, value.y, value.z};
}

static Transform compose_transform(
    const Transform& parent_global,
    const Transform& child_local)
{
    const quat parent_rotation =
        from_raylib_quaternion(parent_global.rotation);
    Transform output = {};
    output.translation = to_raylib_vector(
        from_raylib_vector(parent_global.translation) +
        quat_mul_vec3(
            parent_rotation,
            from_raylib_vector(child_local.translation)));
    output.rotation = to_raylib_quaternion(quat_mul(
        parent_rotation,
        from_raylib_quaternion(child_local.rotation)));
    output.scale = Vector3{1.0f, 1.0f, 1.0f};
    return output;
}

static bool component_near(float first, float second, float tolerance)
{
    return std::fabs(first - second) <= tolerance;
}

static bool quaternion_near(
    Quaternion first,
    Quaternion second,
    float tolerance)
{
    const bool same =
        component_near(first.x, second.x, tolerance) &&
        component_near(first.y, second.y, tolerance) &&
        component_near(first.z, second.z, tolerance) &&
        component_near(first.w, second.w, tolerance);
    const bool opposite =
        component_near(first.x, -second.x, tolerance) &&
        component_near(first.y, -second.y, tolerance) &&
        component_near(first.z, -second.z, tolerance) &&
        component_near(first.w, -second.w, tolerance);
    return same || opposite;
}

static bool transform_near(
    const Transform& first,
    const Transform& second,
    float tolerance)
{
    return
        component_near(
            first.translation.x, second.translation.x, tolerance) &&
        component_near(
            first.translation.y, second.translation.y, tolerance) &&
        component_near(
            first.translation.z, second.translation.z, tolerance) &&
        quaternion_near(first.rotation, second.rotation, tolerance) &&
        component_near(first.scale.x, second.scale.x, tolerance) &&
        component_near(first.scale.y, second.scale.y, tolerance) &&
        component_near(first.scale.z, second.scale.z, tolerance);
}

struct Fixture
{
    BoneInfo bones[G1MeshBoneCount] = {};
    Transform bind_pose[G1MeshBoneCount] = {};
    Transform output[G1MeshBoneCount] = {};
    G1MeshBinding binding = {};
    vec3 accepted_position_values[G1_BoneCount];
    quat accepted_rotation_values[G1_BoneCount];
    slice1d<vec3> accepted_positions;
    slice1d<quat> accepted_rotations;
    int torso_index = 4;
    int head_index = 24;
    char error[256] = {};

    Fixture()
        : accepted_positions(
              G1_BoneCount,
              accepted_position_values),
          accepted_rotations(
              G1_BoneCount,
              accepted_rotation_values)
    {
        for (int model_bone = 0;
             model_bone < G1MeshBoneCount;
             ++model_bone) {
            const int written = std::snprintf(
                bones[model_bone].name,
                sizeof(bones[model_bone].name),
                "%s",
                fixture_names[model_bone]);
            check(
                written > 0 &&
                    written < static_cast<int>(
                        sizeof(bones[model_bone].name)),
                "fixture bone name fits Raylib BoneInfo");
            bones[model_bone].parent = fixture_parents[model_bone];
            bind_pose[model_bone].translation = Vector3{
                0.011f * static_cast<float>(model_bone),
                0.50f + 0.017f * static_cast<float>(model_bone),
                -0.007f * static_cast<float>(model_bone)};
            bind_pose[model_bone].rotation =
                Quaternion{0.0f, 0.0f, 0.0f, 1.0f};
            bind_pose[model_bone].scale = Vector3{1.0f, 1.0f, 1.0f};
            output[model_bone].translation =
                Vector3{101.0f, 102.0f, 103.0f};
            output[model_bone].rotation =
                Quaternion{0.0f, 0.0f, 0.0f, 1.0f};
            output[model_bone].scale = Vector3{1.0f, 1.0f, 1.0f};
        }
        for (int g1_bone = 0; g1_bone < G1_BoneCount; ++g1_bone) {
            accepted_positions(g1_bone) = vec3(
                0.02f * static_cast<float>(g1_bone),
                0.70f + 0.01f * static_cast<float>(g1_bone),
                -0.015f * static_cast<float>(g1_bone));
            accepted_rotations(g1_bone) = quat();
        }
    }
};

static Fixture valid_fixture()
{
    return Fixture();
}

static void build_valid_binding(Fixture& value)
{
    check(
        g1_mesh_binding_build(
            value.binding,
            value.bones,
            value.bind_pose,
            G1MeshBoneCount,
            value.error,
            static_cast<int>(sizeof(value.error))),
        "valid binding builds");
}

static void test_exact_articulated_mapping()
{
    check(g1_mesh_g1_bone_for_name("pelvis") == G1_Hips,
          "pelvis maps to hips");
    check(g1_mesh_g1_bone_for_name("left_ankle_pitch_link") ==
              G1_LeftAnkle,
          "left ankle pitch maps to LeftAnkle");
    check(g1_mesh_g1_bone_for_name("left_ankle_roll_link") == G1_LeftToe,
          "left ankle roll maps to LeftToe");
    check(g1_mesh_g1_bone_for_name("right_ankle_pitch_link") ==
              G1_RightAnkle,
          "right ankle pitch maps to RightAnkle");
    check(g1_mesh_g1_bone_for_name("right_ankle_roll_link") == G1_RightToe,
          "right ankle roll maps to RightToe");
    check(g1_mesh_g1_bone_for_name("imu_in_pelvis") == -1,
          "fixed attachment has no motion bone");
    check(g1_mesh_g1_bone_for_name(nullptr) == -1,
          "null name has no motion bone");

    static const char* const articulated_names[G1_BoneCount] = {
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
    for (int g1_bone = G1_Hips;
         g1_bone <= G1_RightWrist;
         ++g1_bone) {
        check(
            g1_mesh_g1_bone_for_name(articulated_names[g1_bone]) ==
                g1_bone,
            "every articulated name maps to its exact G1 bone");
    }

    Fixture value = valid_fixture();
    build_valid_binding(value);
    int counts[G1_BoneCount] = {};
    for (int model_bone = 0;
         model_bone < G1MeshBoneCount;
         ++model_bone) {
        const int g1_bone = value.binding.model_to_g1[model_bone];
        if (g1_bone >= 0) ++counts[g1_bone];
    }
    check(counts[G1_Simulation] == 0,
          "simulation has no model bone");
    for (int g1_bone = G1_Hips;
         g1_bone <= G1_RightWrist;
         ++g1_bone) {
        check(counts[g1_bone] == 1,
              "every articulated G1 bone maps exactly once");
    }
}

static void test_fixed_attachment_follows_parent_bind_local()
{
    Fixture value = valid_fixture();
    build_valid_binding(value);
    value.accepted_positions(G1_Spine2) = vec3(1.0f, 2.0f, 3.0f);
    value.accepted_rotations(G1_Spine2) =
        quat_from_angle_axis(0.5f, vec3(0.0f, 1.0f, 0.0f));
    check(g1_mesh_pose_build(
              value.output, G1MeshBoneCount, value.binding,
              value.accepted_positions, value.accepted_rotations,
              value.error, sizeof(value.error)),
          "finite accepted pose builds");
    check(transform_near(
              value.output[value.head_index],
              compose_transform(
                  value.output[value.torso_index],
                  value.binding.bind_local[value.head_index]),
              5.0e-6f),
          "head preserves authored torso-local bind transform");
}

static void test_rest_pose_reproduces_global_bind_pose()
{
    Fixture value = valid_fixture();
    build_valid_binding(value);
    for (int model_bone = 0;
         model_bone < G1MeshBoneCount;
         ++model_bone) {
        const int g1_bone = value.binding.model_to_g1[model_bone];
        if (g1_bone < 0) continue;
        value.accepted_positions(g1_bone) =
            from_raylib_vector(value.bind_pose[model_bone].translation);
        value.accepted_rotations(g1_bone) =
            from_raylib_quaternion(value.bind_pose[model_bone].rotation);
    }
    check(
        g1_mesh_pose_build(
            value.output,
            G1MeshBoneCount,
            value.binding,
            value.accepted_positions,
            value.accepted_rotations,
            value.error,
            static_cast<int>(sizeof(value.error))),
        "rest pose builds");
    for (int model_bone = 0;
         model_bone < G1MeshBoneCount;
         ++model_bone) {
        check(
            transform_near(
                value.output[model_bone],
                value.bind_pose[model_bone],
                0.0005f),
            "rest transfer reproduces every global bind transform");
    }
}

static void test_binding_rejects_malformed_layouts()
{
    {
        Fixture value = valid_fixture();
        value.bones[33].parent = 0;
        check(!g1_mesh_binding_build(
                  value.binding, value.bones, value.bind_pose,
                  G1MeshBoneCount, value.error, sizeof(value.error)),
              "mapped parent mismatch is rejected");
    }
    {
        Fixture value = valid_fixture();
        std::snprintf(
            value.bones[32].name,
            sizeof(value.bones[32].name),
            "%s",
            "missing_left_hip");
        check(!g1_mesh_binding_build(
                  value.binding, value.bones, value.bind_pose,
                  G1MeshBoneCount, value.error, sizeof(value.error)),
              "missing mapped bone is rejected");
    }
    {
        Fixture value = valid_fixture();
        std::snprintf(
            value.bones[33].name,
            sizeof(value.bones[33].name),
            "%s",
            value.bones[32].name);
        check(!g1_mesh_binding_build(
                  value.binding, value.bones, value.bind_pose,
                  G1MeshBoneCount, value.error, sizeof(value.error)),
              "duplicate mapped bone is rejected");
    }
    {
        Fixture value = valid_fixture();
        check(!g1_mesh_binding_build(
                  value.binding, value.bones, value.bind_pose,
                  G1MeshBoneCount - 1, value.error, sizeof(value.error)),
              "wrong model bone count is rejected");
    }
}

static void test_pose_rejects_invalid_inputs_transactionally()
{
    {
        Fixture value = valid_fixture();
        build_valid_binding(value);
        Transform before[G1MeshBoneCount] = {};
        std::memcpy(before, value.output, sizeof(before));
        value.accepted_positions(G1_LeftToe).x =
            std::numeric_limits<float>::quiet_NaN();
        check(!g1_mesh_pose_build(
                  value.output, G1MeshBoneCount, value.binding,
                  value.accepted_positions, value.accepted_rotations,
                  value.error, sizeof(value.error)),
              "NaN accepted position is rejected");
        check(std::memcmp(before, value.output, sizeof(before)) == 0,
              "NaN failure does not partially publish output");
    }
    {
        Fixture value = valid_fixture();
        build_valid_binding(value);
        value.accepted_rotations(G1_RightWrist) =
            quat(2.0f, 0.0f, 0.0f, 0.0f);
        check(!g1_mesh_pose_build(
                  value.output, G1MeshBoneCount, value.binding,
                  value.accepted_positions, value.accepted_rotations,
                  value.error, sizeof(value.error)),
              "non-unit accepted quaternion is rejected");
    }
    {
        Fixture value = valid_fixture();
        build_valid_binding(value);
        check(!g1_mesh_pose_build(
                  value.output, G1MeshBoneCount - 1, value.binding,
                  value.accepted_positions, value.accepted_rotations,
                  value.error, sizeof(value.error)),
              "wrong output count is rejected");
        check(!g1_mesh_pose_build(
                  value.output, G1MeshBoneCount, value.binding,
                  slice1d<vec3>(
                      G1_BoneCount - 1,
                      value.accepted_position_values),
                  value.accepted_rotations,
                  value.error, sizeof(value.error)),
              "wrong position count is rejected");
        check(!g1_mesh_pose_build(
                  value.output, G1MeshBoneCount, value.binding,
                  value.accepted_positions,
                  slice1d<quat>(
                      G1_BoneCount - 1,
                      value.accepted_rotation_values),
                  value.error, sizeof(value.error)),
              "wrong rotation count is rejected");
    }
    {
        Fixture value = valid_fixture();
        build_valid_binding(value);
        Transform* const aliased_output =
            reinterpret_cast<Transform*>(value.accepted_position_values);
        check(!g1_mesh_pose_build(
                  aliased_output, G1MeshBoneCount, value.binding,
                  value.accepted_positions, value.accepted_rotations,
                  value.error, sizeof(value.error)),
              "output alias with accepted positions is rejected");
    }
}

static void test_unload_is_idempotent_for_canonical_renderer()
{
    G1MeshRenderer renderer = {};
    g1_mesh_renderer_unload(renderer);
    g1_mesh_renderer_unload(renderer);
    check(!renderer.loaded, "second canonical unload remains unloaded");
    check(renderer.model.meshCount == 0 && renderer.model.meshes == nullptr,
          "second canonical unload preserves empty model");
    check(renderer.animation.frameCount == 0 &&
              renderer.animation.framePoses == nullptr &&
              renderer.animation.bones == nullptr,
          "second canonical unload preserves empty animation");
}

int main()
{
    test_exact_articulated_mapping();
    test_fixed_attachment_follows_parent_bind_local();
    test_rest_pose_reproduces_global_bind_pose();
    test_binding_rejects_malformed_layouts();
    test_pose_rejects_invalid_inputs_transactionally();
    test_unload_is_idempotent_for_canonical_renderer();
    return 0;
}
