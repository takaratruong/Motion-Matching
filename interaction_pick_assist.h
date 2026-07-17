#pragma once

#include "interaction_arrival.h"
#include "interaction_pick_approach.h"
#include "interaction_pick_slots.h"

#include <array>
#include <cstdint>
#include <optional>

namespace interaction {

enum class PickAssistState : uint8_t {
    Idle,
    CoarseApproach,
    Preview,
    FinalApproach,
    Settling,
    FinalPreview,
    ReadyToSubmit,
    Submitted,
    Failed,
    SlotApproach,
};

enum class PickAssistReason : uint8_t {
    None,
    Cancelled,
    TargetUnavailable,
    TargetChanged,
    RuntimeChanged,
    OutsideTravelEnvelope,
    NoFeasibleEntry,
    PreviewDeadline,
    ArrivalDeadline,
    FinalPreviewRejected,
    SlotChanged,
    NoAuthoredSlot,
    InvalidGeometry,
    TableBlocked,
    ObstacleBlocked,
    AllSlotsBlocked,
    PoorMatch,
};

struct PickAssistConfig {
    float maximum_assisted_path_m = 1.00F;
    float reach_entry_distance_m = 0.60F;
    float reach_entry_tolerance_m = 0.12F;
    float maximum_settle_position_error_m = 0.15F;
    float maximum_settle_displayed_speed_mps = 0.10F;
    uint32_t required_settle_ticks = 5U;
    uint32_t maximum_preview_ticks = 250U;
    uint32_t maximum_arrival_ticks = 250U;
    ArrivalConfig arrival{};
};

struct PickAssistStart {
    InteractionTarget target_snapshot{};
    std::vector<PickNavigationObstacle> obstacles{};

    // Temporary compatibility surface for the legacy two-slot funnel tests.
    TargetHandle target{};
    uint32_t affordance_id = 0U;
    Hand hand = Hand::Right;
    Transform object_world{};
    Transform root_world{};
    Transform reach_waypoint{};
    PickEntrySlots slots{};
};

struct PickAssistObservation {
    RuntimeState runtime_state = RuntimeState::Locomotion;
    const InteractionTarget* target = nullptr;
    Transform displayed_root{};
    vec3 simulation_velocity{};
    float displayed_planar_speed_mps = 0.0F;
    float camera_azimuth = 0.0F;
    uint64_t snapshot_fingerprint = 0U;
    uint64_t preview_snapshot_fingerprint = 0U;
    std::optional<std::array<PickEntryPreview, 2>> previews{};
};

struct PickAssistOutput {
    bool override_steering = false;
    vec3 left_stick{};
    vec3 right_stick{};
    bool force_strafe = false;
    bool stationary_constraint = false;
    bool needs_preview = false;
    bool submit_interact = false;
};

struct PickAssistFinalPreviewDiagnostics {
    bool available = false;
    bool all_preview_roots_finite = false;
    bool fingerprint_equal = false;
    bool path_feasible = false;
    Reason path_reason = Reason::None;
    bool match_ready = false;
    Reason match_reason = Reason::None;
    bool prospective_root_equal = false;
    int32_t feasible_entry_frame = -1;
    int32_t contact_frame = -1;
    float total_cost = 0.0F;
};

struct PickAssistDiagnostics {
    PickAssistState state = PickAssistState::Idle;
    PickAssistReason reason = PickAssistReason::None;
    TargetHandle target{};
    uint32_t affordance_id = 0U;
    int selected_slot = -1;
    uint32_t selected_slot_id = 0U;
    PickSlotSelection slot_selection{};
    uint32_t settle_ticks = 0U;
    std::array<float, 2> slot_route_lengths_m{};
    std::array<bool, 2> slot_permitted{};
    float route_length_m = 0.0F;
    float assisted_travel_m = 0.0F;
    float object_origin_distance_m = 0.0F;
    float object_bounds_center_distance_m = 0.0F;
    float root_error_m = 0.0F;
    float yaw_error_radians = 0.0F;
    float speed_mps = 0.0F;
    PickAssistFinalPreviewDiagnostics final_preview{};
};

class ControllerPickAssist {
public:
    explicit ControllerPickAssist(PickAssistConfig config = {});
    bool begin(const PickAssistStart& start);
    bool begin(
        const PickAssistStart& start,
        const InteractionTarget* post_step_target);
    void cancel();
    PickAssistOutput observe(const PickAssistObservation& observation);
    std::optional<PickRequest> take_submission(uint64_t request_id);
    bool active() const;
    bool owns_manual_interact() const;
    const PickAssistDiagnostics& diagnostics() const;

private:
    PickAssistConfig config_{};
    PickAssistStart start_{};
    MappedPickSlot frozen_slot_{};
    bool frozen_slot_attempt_ = false;
    Transform previous_observed_root_{};
    vec3 entry_point_{};
    uint32_t preview_ticks_ = 0U;
    uint32_t arrival_ticks_ = 0U;
    PickAssistDiagnostics diagnostics_{};
};

const char* pick_assist_state_name(PickAssistState state);
const char* pick_assist_reason_name(PickAssistReason reason);
PickAssistReason pick_assist_reason_from_slot_reason(PickSlotReason reason);

}  // namespace interaction
