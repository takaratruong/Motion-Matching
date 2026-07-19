#pragma once

#include "interaction_attachment.h"
#include "interaction_carry.h"
#include "interaction_pickup_provenance.h"
#include "interaction_place_controller.h"
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
    PlacePreflight,
    PlaceAlign,
    PlaceReplay,
    PlaceRelease,
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
    PlaceControllerConfig place{};
};

struct PickEntryRoot {
    float world_x = 0.0F;
    float world_z = 0.0F;
    float world_yaw_radians = 0.0F;
};

struct PickEntryPreview {
    bool path_feasible = false;
    bool match_ready = false;
    Reason path_reason = Reason::None;
    Reason match_reason = Reason::None;
    PickEntryRoot prospective_root{};
    int32_t feasible_entry_frame = -1;
    int32_t contact_frame = -1;
    float total_cost = 0.0F;
    MatchCandidate match_candidate{};
    CertifiedPickupSourceIdentity pickup_source{};
};

struct RuntimeInput {
    float dt = 0.0F;
    LocomotionSnapshot locomotion{};
    bool interact_pressed = false;
    std::optional<PickRequest> pick_request{};
    std::optional<PlaceRequest> place_request{};
    bool cancel_pressed = false;
    bool reset_pressed = false;
};

struct RuntimePlaceDiagnostics {
    SurfaceHandle surface{};
    uint32_t affordance_id = 0U;
    PlaceMotionMode mode = PlaceMotionMode::None;
    uint64_t selection_id = 0U;
    bool preview_available = false;
    PlaceStagingPreview preview{};
    bool candidate_certified = false;
    bool preflight_config_identity = false;
    IKConfig effective_ik{};
    uint64_t ik_config_fingerprint = 0U;
    Transform requested_goal_world{};
    PlacementFit requested_fit{};
    PlacementFit actual_fit{};
    int32_t clip = -1;
    int32_t source_frame = -1;
    double source_frame_exact = -1.0;
    int32_t commit_frame = -1;
    PlacePhase phase = PlacePhase::Align;
    float time_to_release_seconds = 0.0F;
    bool committed = false;
    bool release_due = false;
    bool released = false;
    bool support_sweep_clear = false;
    float support_position_error_m = 0.0F;
    float support_orientation_error_radians = 0.0F;
    float requested_root_correction_m = 0.0F;
    float applied_root_correction_m = 0.0F;
    float requested_yaw_correction_radians = 0.0F;
    float applied_yaw_correction_radians = 0.0F;
    float requested_hand_correction_m = 0.0F;
    float applied_hand_correction_m = 0.0F;
    float requested_hand_orientation_radians = 0.0F;
    float applied_hand_orientation_radians = 0.0F;
    Reason reason = Reason::None;
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
    bool inactive_arm_targets_locomotion = false;
    bool inactive_arm_tracks_locomotion = false;
    bool pack_available = false;
    CertifiedPickupSourceIdentity pickup_source{};
    RuntimePlaceDiagnostics place{};
};

struct RuntimeOutput {
    bool owns_pose = false;
    bool suppress_steering = false;
    Pose pose{};
    Transform object_world{};
    RuntimeDiagnostics diagnostics{};
};

namespace runtime_detail {

struct RealizedPickTransitionEvaluation {
    bool feasible = false;
    Reason reason = Reason::None;
    int32_t frame = -1;
};

Reason realized_pick_hard_rejection_reason(Reason reason);

RealizedPickTransitionEvaluation evaluate_realized_pick_transition(
    const Database& database,
    const Pose& live_entry_pose,
    const MatchCandidate& candidate,
    const InteractionTarget& target,
    const GraspAffordance& affordance,
    const PlaybackConfig& playback_config,
    const IKConfig& ik_config,
    const AttachmentConfig& attachment_config = AttachmentConfig{});

struct PickSnapshotMap {
    bool accepted = false;
    Reason reason = Reason::None;
    LocomotionSnapshot snapshot{};
};

PickSnapshotMap map_pick_entry_snapshot(
    const LocomotionSnapshot& snapshot,
    PickEntryRoot root);
uint64_t locomotion_snapshot_fingerprint(
    const LocomotionSnapshot& snapshot);

}  // namespace runtime_detail

class InteractionRuntime {
public:
    InteractionRuntime(
        const Database& database,
        const Features& features,
        TargetRegistry& registry,
        RuntimeConfig config,
        const CertifiedPickupSourceRegistry* pickup_source_registry =
            nullptr);
    InteractionRuntime(
        const Database& database,
        const Features& features,
        TargetRegistry& registry,
        PlacementSurfaceRegistry& surface_registry,
        const PlaceMotionLibrary& place_library,
        RuntimeConfig config,
        const CertifiedPickupSourceRegistry* pickup_source_registry =
            nullptr);
    static InteractionRuntime disabled(Reason reason);
    RuntimeState state() const;
    const RuntimeDiagnostics& diagnostics() const;
    PickEntryPreview preview_pick(
        const LocomotionSnapshot& live_flat_snapshot,
        PickEntryRoot prospective_root,
        TargetHandle target,
        uint32_t affordance_id) const;
    PlaceStagingPreview preview_place(
        SurfaceHandle surface,
        uint32_t affordance_id) const;
    RuntimeOutput update(const RuntimeInput& input);

private:
    friend struct InteractionRuntimeTestAccess;

    struct PickEvaluationBuild {
        bool accepted = false;
        Reason reason = Reason::None;
        matcher_detail::PickEvaluationInput input{};
    };

    struct PlaceMatchBuildResult {
        bool accepted = false;
        Reason reason = Reason::None;
        PlaceMatchInput input{};
    };

    InteractionRuntime() = default;
    PickEvaluationBuild build_pick_evaluation(
        const LocomotionSnapshot& locomotion,
        TargetHandle target_handle,
        uint32_t affordance_id,
        bool require_free) const;
    matcher_detail::PickEvaluation evaluate_pick_entries_realized(
        const matcher_detail::PickEvaluationInput& input) const;
    void advance_pick_clearance(float target_elapsed_seconds);
    void drain_playback_events(float published_elapsed_seconds);
    void begin_carry();
    PlaceMatchBuildResult make_place_match_input(
        SurfaceHandle surface,
        uint32_t affordance_id) const;
    void reconstruct_carry(
        const Pose& pose,
        Transform object_world,
        Reason reason,
        ResultCode result);
    void reset_place_attempt();
    void update_place_diagnostics(const PlaceStep& step);

    const Database* database_ = nullptr;
    const Features* features_ = nullptr;
    TargetRegistry* registry_ = nullptr;
    PlacementSurfaceRegistry* surface_registry_ = nullptr;
    const PlaceMotionLibrary* place_library_ = nullptr;
    const CertifiedPickupSourceRegistry* pickup_source_registry_ = nullptr;
    RuntimeConfig config_{};
    std::optional<PickRequest> request_{};
    std::optional<InteractionTarget> target_{};
    std::optional<GraspAffordance> affordance_{};
    std::optional<MatchCandidate> candidate_{};
    std::optional<SequentialPlayer> player_{};
    std::optional<SequentialPlayer> event_player_{};
    std::optional<SequentialPlayer> clearance_player_{};
    std::optional<AttachmentController> attachment_{};
    std::optional<CarryController> carry_{};
    std::optional<PlaceController> place_controller_{};
    std::optional<PlaceRequest> place_request_{};
    std::optional<PlaceMatchInput> frozen_place_input_{};
    std::optional<PlaceMatchInput> active_place_input_{};
    PlaceStagingPreview frozen_place_preview_{};
    PlaceStep place_step_{};
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
    bool place_final_frame_presented_ = false;
    Reason post_commit_reason_ = Reason::None;
    RuntimeState state_ = RuntimeState::Locomotion;
    RuntimeDiagnostics diagnostics_{};
};

}  // namespace interaction
