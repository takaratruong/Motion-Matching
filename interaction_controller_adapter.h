#pragma once

#include "interaction_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

namespace interaction {

inline constexpr float kControllerStepSeconds = 1.0F / 60.0F;
inline constexpr float kInteractionRuntimeStepSeconds = 1.0F / 25.0F;
inline constexpr std::array<float, 14> kFlatControllerRestHandDof{};
inline constexpr std::array<float, 14>
    kFlatControllerRestHandDofVelocities{};

inline constexpr size_t kFlatControllerBoneCount = 23U;
inline constexpr size_t kFlatControllerLeftToe = 5U;
inline constexpr size_t kFlatControllerRightToe = 9U;

inline constexpr std::array<int32_t, kFlatControllerBoneCount>
    kFlatControllerParents = {
        -1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10,
        11, 12, 13, 12, 15, 16, 17, 12, 19, 20, 21};

struct FlatControllerAnchor {
    size_t flat_bone;
    size_t g1_bone;
};

inline constexpr std::array<FlatControllerAnchor, 21>
    kFlatControllerAnchors = {{
        {0U, 0U},
        {1U, 1U},
        {2U, 2U},
        {3U, 5U},
        {4U, 6U},
        {5U, 7U},
        {6U, 8U},
        {7U, 11U},
        {8U, 12U},
        {9U, 13U},
        {10U, 14U},
        {11U, 15U},
        {12U, 16U},
        {15U, 17U},
        {16U, 19U},
        {17U, 20U},
        {18U, 23U},
        {19U, 24U},
        {20U, 26U},
        {21U, 27U},
        {22U, 30U},
    }};

struct FlatControllerPose {
    std::array<vec3, kFlatControllerBoneCount> positions{};
    std::array<vec3, kFlatControllerBoneCount> velocities{};
    std::array<quat, kFlatControllerBoneCount> rotations{};
    std::array<vec3, kFlatControllerBoneCount> angular_velocities{};
    std::array<uint8_t, 2> foot_contacts{};
};

Pose expand_flat_controller_pose(
    const FlatControllerPose& flat_pose,
    const Pose& interaction_reference);
FlatControllerPose collapse_interaction_pose(
    const Pose& interaction_pose,
    const FlatControllerPose& flat_fallback);
FlatControllerPose collapse_interaction_pose(
    const Pose& interaction_pose,
    const Pose& interaction_reference,
    const FlatControllerPose& flat_reference);

struct ControllerInteractionEdges {
    bool interact_pressed = false;
    bool cancel_pressed = false;
    bool reset_pressed = false;
};

class ControllerInteractionScheduler {
public:
    using LocomotionProvider = std::function<LocomotionSnapshot()>;
    using PickRequestResolver = std::function<
        std::optional<PickRequest>(const LocomotionSnapshot&)>;
    using RuntimeUpdate = std::function<RuntimeOutput(const RuntimeInput&)>;

    const RuntimeOutput& tick(
        ControllerInteractionEdges edges,
        const LocomotionProvider& locomotion_provider,
        const PickRequestResolver& request_resolver,
        const RuntimeUpdate& runtime_update);

    int phase() const;
    const RuntimeOutput& cached_output() const;

private:
    int phase_ = 0;
    bool pending_interact_ = false;
    bool pending_cancel_ = false;
    bool pending_reset_ = false;
    RuntimeOutput cached_output_{};
};

struct ControllerInteractionFrameState {
    FlatControllerPose pose{};
    bool runtime_owns_pose = false;
    bool overrides_locomotion_pose = false;
    bool synchronize_simulation_root = false;
    vec3 simulation_root_position{};
    quat simulation_root_rotation{};
};

class ControllerInteractionFrameHandoff {
public:
    ControllerInteractionFrameState apply(
        const FlatControllerPose& locomotion_pose,
        const RuntimeOutput& runtime_output,
        float dt);
    void reset();

private:
    bool runtime_owned_last_update_ = false;
    bool release_active_ = false;
    float blend_seconds_ = 0.0F;
    FlatControllerPose blend_source_{};
    FlatControllerPose ownership_fallback_{};
    FlatControllerPose last_rendered_pose_{};
};

struct ControllerInteractionSceneState {
    Transform object_world{};
    bool runtime_authority = false;
};

class ControllerInteractionSceneHandoff {
public:
    ControllerInteractionSceneState apply(
        const InteractionTarget* registry_target,
        const RuntimeOutput& runtime_output,
        const Transform& authored_fallback);

private:
    bool has_runtime_pose_ = false;
    TargetHandle runtime_target_{};
    Transform runtime_object_world_{};
};

const char* controller_carry_mode_label(const RuntimeOutput& output);

InteractionTarget make_controller_demo_target(const Database& database);
void validate_controller_interaction_pack(
    const Database& database,
    const Features& features);

}  // namespace interaction
