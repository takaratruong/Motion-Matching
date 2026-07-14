#pragma once

#include "interaction_runtime.h"

#include <array>
#include <functional>
#include <optional>

namespace interaction {

inline constexpr float kControllerStepSeconds = 1.0F / 60.0F;
inline constexpr float kInteractionRuntimeStepSeconds = 1.0F / 25.0F;
inline constexpr std::array<float, 14> kFlatControllerRestHandDof{};
inline constexpr std::array<float, 14>
    kFlatControllerRestHandDofVelocities{};

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

class ControllerInteractionAdapter {
public:
    Pose apply(
        const Pose& locomotion_pose,
        const RuntimeOutput& runtime_output,
        float dt);
    void reset();

private:
    bool owned_last_update_ = false;
    float blend_seconds_ = 0.0F;
    Pose blend_source_{};
};

struct ControllerInteractionFrameState {
    Pose pose{};
    bool owns_pose = false;
    bool synchronize_simulation_root = false;
    vec3 simulation_root_position{};
    quat simulation_root_rotation{};
};

class ControllerInteractionFrameHandoff {
public:
    ControllerInteractionFrameState apply(
        const Pose& locomotion_pose,
        const RuntimeOutput& runtime_output,
        float dt);
    void reset();

private:
    ControllerInteractionAdapter adapter_{};
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
