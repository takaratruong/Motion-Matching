#pragma once

#include "interaction_attachment.h"
#include "interaction_carry.h"
#include "interaction_playback.h"

#include <array>
#include <cstdint>
#include <optional>

namespace interaction {

enum class RuntimeState : uint8_t {
    Disabled,
    Locomotion,
    Preflight,
    Align,
    PickupReplay,
    Hold,
    Carry,
};

struct PlaybackConfig {
    float canonical_fps = 25.0F;
    float speed = 1.0F;
    float minimum_speed = 0.85F;
    float maximum_speed = 1.15F;
    float entry_blend_seconds = 0.25F;
    float maximum_alignment_seconds = 1.00F;
    float commit_horizon_seconds = 0.50F;
};

struct RuntimeConfig {
    MatchConfig matcher{};
    PlaybackConfig playback{};
    IKConfig ik{};
    AttachmentConfig attachment{};
    CarryConfig carry{};
};

struct RuntimeInput {
    float dt = 0.0F;
    LocomotionSnapshot locomotion{};
    bool interact_pressed = false;
    std::optional<PickRequest> pick_request{};
    bool cancel_pressed = false;
    bool reset_pressed = false;
};

struct RuntimeDiagnostics {
    RuntimeState state = RuntimeState::Disabled;
    ResultCode result = ResultCode::None;
    Reason reason = Reason::None;
    TargetHandle target{};
    ObjectState object_state = ObjectState::Free;
    uint32_t affordance_id = 0;
    int32_t clip = -1;
    int32_t frame = -1;
    Phase phase = Phase::Approach;
    Hand hand = Hand::Right;
    float total_cost = 0.0F;
    std::array<float, 5> group_costs{};
    float requested_root_correction_m = 0.0F;
    float applied_root_correction_m = 0.0F;
    float requested_yaw_correction_radians = 0.0F;
    float applied_yaw_correction_radians = 0.0F;
    float playback_speed = 1.0F;
    float hand_position_error_m = 0.0F;
    float hand_orientation_error_radians = 0.0F;
    float hand_constraint_weight = 0.0F;
    bool attached = false;
    bool recorded_carry = false;
    bool inactive_arm_tracks_locomotion = false;
    bool pack_available = false;
};

struct RuntimeOutput {
    bool owns_pose = false;
    bool suppress_steering = false;
    Pose pose{};
    Transform object_world{};
    RuntimeDiagnostics diagnostics{};
};

class InteractionRuntime {
public:
    InteractionRuntime(
        const Database& database,
        const Features& features,
        TargetRegistry& registry,
        RuntimeConfig config);
    static InteractionRuntime disabled(Reason reason);
    RuntimeState state() const;
    const RuntimeDiagnostics& diagnostics() const;
    RuntimeOutput update(const RuntimeInput& input);

private:
    InteractionRuntime() = default;
    void drain_playback_events(float published_elapsed_seconds);
    void begin_carry();

    const Database* database_ = nullptr;
    const Features* features_ = nullptr;
    TargetRegistry* registry_ = nullptr;
    RuntimeConfig config_{};
    std::optional<PickRequest> request_{};
    std::optional<InteractionTarget> target_{};
    std::optional<GraspAffordance> affordance_{};
    std::optional<MatchCandidate> candidate_{};
    std::optional<SequentialPlayer> player_{};
    std::optional<SequentialPlayer> event_player_{};
    std::optional<AttachmentController> attachment_{};
    std::optional<CarryController> carry_{};
    Pose entry_blend_source_{};
    Pose pose_{};
    Transform target_hand_world_{};
    Transform source_contact_hand_world_{};
    Transform original_object_world_{};
    Transform object_world_{};
    vec3 previous_corrected_hand_world_{};
    float entry_blend_elapsed_seconds_ = 0.0F;
    float commit_seconds_ = 0.0F;
    bool owns_reservation_ = false;
    bool has_previous_corrected_hand_ = false;
    bool corrected_path_clear_ = true;
    bool contact_evaluated_ = false;
    bool ever_attached_ = false;
    bool post_commit_failure_ = false;
    bool final_failure_frame_presented_ = false;
    bool carry_ready_ = false;
    bool carry_started_ = false;
    Reason post_commit_reason_ = Reason::None;
    RuntimeState state_ = RuntimeState::Locomotion;
    RuntimeDiagnostics diagnostics_{};
};

}  // namespace interaction
