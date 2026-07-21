#pragma once

#include "interaction_runtime.h"
#include "interaction_target_rig_ik.h"
#include "locomotion_timing.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

namespace interaction {

inline constexpr float kControllerStepSeconds =
    locomotion_timing::kStepSeconds;
inline constexpr float kInteractionRuntimeStepSeconds =
    locomotion_timing::kStepSeconds;
inline constexpr float kPlaceStagingMaximumRootErrorM = 0.25F;
inline constexpr float kPlaceStagingMaximumYawErrorRadians = 0.436332313F;
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
// Builds a rendering-only G1 reference whose mapped world rotations follow the
// flat locomotion reference while retaining the authored G1 bone geometry.
Pose calibrate_flat_mesh_reference(
    const Pose& geometry_reference,
    const FlatControllerPose& flat_reference);
// Expands the representable flat-controller subspace around a fixed reference
// pair. Root translation/velocity and mapped world rotation/angular-velocity
// deltas are transferred; true-G1 non-root translation/velocity morphology and
// unmapped local channels are invariant. Unsupported flat non-root translation
// and velocity deltas are therefore projected out.
Pose expand_flat_controller_pose(
    const FlatControllerPose& flat_pose,
    const Pose& interaction_reference,
    const FlatControllerPose& flat_reference);
FlatControllerPose collapse_interaction_pose(
    const Pose& interaction_pose,
    const Pose& interaction_reference,
    const FlatControllerPose& flat_reference);

struct ControllerInteractionEdges {
    bool interact_pressed = false;
    bool cancel_pressed = false;
    bool reset_pressed = false;
};

struct ControllerPlaceTarget {
    SurfaceHandle surface{};
    uint32_t affordance_id = 0U;
    uint64_t request_id = 0U;
};

class ControllerInteractionScheduler {
public:
    using LocomotionProvider = std::function<LocomotionSnapshot()>;
    using PickRequestResolver = std::function<
        std::optional<PickRequest>(const LocomotionSnapshot&)>;
    using PlaceTargetResolver = std::function<
        std::optional<ControllerPlaceTarget>(const LocomotionSnapshot&)>;
    using PlacePreviewResolver = std::function<
        PlaceStagingPreview(SurfaceHandle, uint32_t)>;
    using RuntimeUpdate = std::function<RuntimeOutput(const RuntimeInput&)>;

    const RuntimeOutput& tick(
        ControllerInteractionEdges edges,
        const LocomotionProvider& locomotion_provider,
        const PickRequestResolver& request_resolver,
        const RuntimeUpdate& runtime_update);
    const RuntimeOutput& tick(
        ControllerInteractionEdges edges,
        const LocomotionProvider& locomotion_provider,
        const PickRequestResolver& request_resolver,
        const PlaceTargetResolver& place_target_resolver,
        const PlacePreviewResolver& place_preview_resolver,
        const RuntimeUpdate& runtime_update);

    int phase() const;
    bool updated_last_tick() const;
    const RuntimeOutput& cached_output() const;
    const std::optional<ControllerPlaceTarget>& latched_place() const;
    const std::optional<PlaceStagingPreview>& place_preview() const;

private:
    int phase_ = 0;
    bool pending_interact_ = false;
    bool pending_cancel_ = false;
    bool pending_reset_ = false;
    bool updated_last_tick_ = false;
    RuntimeOutput cached_output_{};
    std::optional<ControllerPlaceTarget> latched_place_{};
    std::optional<PlaceStagingPreview> place_preview_{};
    bool place_submitted_ = false;
};

struct ControllerInteractionFrameState {
    FlatControllerPose pose{};
    TargetRigArmIKResult hand_constraint_result{};
    bool hand_constraint_validated = false;
    quat hand_constraint_calibration_rotation{};
    bool runtime_owns_pose = false;
    bool overrides_locomotion_pose = false;
    bool synchronize_simulation_root = false;
    vec3 simulation_root_position{};
    quat simulation_root_rotation{};
};

struct ControllerInteractionHandConstraint {
    TargetHandle target{};
    uint32_t affordance_id = 0U;
    Hand hand = Hand::Right;
    Transform grasp_world{};
};

class ControllerInteractionFrameHandoff {
public:
    ControllerInteractionFrameState apply(
        const FlatControllerPose& locomotion_pose,
        const RuntimeOutput& runtime_output,
        float dt,
        std::optional<ControllerInteractionHandConstraint> hand_constraint =
            std::nullopt);
    void reset();

private:
    bool runtime_owned_last_update_ = false;
    bool release_active_ = false;
    float blend_seconds_ = 0.0F;
    FlatControllerPose blend_source_{};
    Pose ownership_interaction_reference_{};
    FlatControllerPose ownership_flat_reference_{};
    FlatControllerPose last_rendered_pose_{};
    TargetHandle ownership_target_{};
    uint32_t ownership_affordance_id_ = 0U;
    Hand ownership_hand_ = Hand::Right;
    bool ownership_identity_poisoned_ = false;
    TargetRigArmIK target_rig_arm_ik_{};
    std::optional<ControllerInteractionHandConstraint>
        ownership_hand_constraint_{};
    bool hand_constraint_applied_last_update_ = false;
    std::optional<Hand> active_arm_release_hand_{};
    float active_arm_release_blend_seconds_ = 0.0F;
    FlatControllerPose active_arm_release_source_{};
    std::optional<Hand> post_release_active_arm_hand_{};
    float post_release_active_arm_blend_seconds_ = 0.0F;
    FlatControllerPose post_release_active_arm_source_{};
    std::optional<Hand> inactive_arm_locomotion_hand_{};
    std::optional<Hand> inactive_arm_return_hand_{};
    float inactive_arm_blend_seconds_ = 0.0F;
    FlatControllerPose inactive_arm_blend_source_{};
    bool layered_carry_last_update_ = false;
    bool lower_body_inertialization_active_ = false;
    float lower_body_inertialization_seconds_ = 0.0F;
    FlatControllerPose lower_body_inertial_offsets_{};
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
        const Transform& authored_fallback,
        float alpha,
        bool runtime_sample_updated);

private:
    void reset_authority();

    bool has_runtime_pose_ = false;
    TargetHandle runtime_target_{};
    Transform current_runtime_object_world_{};
};

const char* controller_carry_mode_label(const RuntimeOutput& output);
RuntimePlaceDiagnostics controller_place_debug_diagnostics(
    const RuntimeOutput& output,
    const std::optional<PlaceStagingPreview>& staged_preview);

InteractionTarget make_controller_demo_target(const Database& database);
PlacementSurface make_controller_demo_destination_surface(
    const Database& database,
    const InteractionTarget& source_target);
void validate_controller_interaction_pack(
    const Database& database,
    const Features& features);

}  // namespace interaction
