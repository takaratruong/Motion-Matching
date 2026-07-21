#pragma once

#include "interaction_arrival.h"
#include "interaction_pick_slots.h"
#include "interaction_runtime.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace interaction {

enum class PickAssistState : uint8_t {
    Idle,
    SlotApproach,
    Settling,
    FinalPreview,
    ReadyToSubmit,
    Submitted,
    Failed,
    SlotSelectionPreview = 7,
};

enum class PickAssistReason : uint8_t {
    None,
    Cancelled,
    TargetUnavailable,
    TargetChanged,
    SlotChanged,
    RuntimeChanged,
    NoAuthoredSlot,
    InvalidGeometry,
    OutsideTravelEnvelope,
    TableBlocked,
    ObstacleBlocked,
    AllSlotsBlocked,
    ArrivalDeadline,
    PoorMatch,
    FinalPreviewRejected,
    SelectionPreviewRejected = 15,
};

struct PickAssistConfig {
    float maximum_assisted_path_m = 1.00F;
    float maximum_settle_position_error_m = 0.15F;
    float maximum_settle_displayed_speed_mps = 0.10F;
    uint32_t required_settle_ticks = 5U;
    uint32_t maximum_arrival_ticks = 250U;
    ArrivalConfig arrival{};
};

struct PickAssistStart {
    InteractionTarget target_snapshot{};
    uint32_t affordance_id = 0U;
    Transform root_world{};
    std::vector<PickNavigationObstacle> obstacles{};
    uint64_t controller_tick = 0U;
    std::vector<PickNavigationObstacle> live_obstacles{};
};

struct PickAssistPreviewRequest {
    uint32_t slot_id = 0U;
    PickEntryRoot root{};
};

struct PickAssistPreviewResult {
    PickAssistPreviewRequest request{};
    std::optional<PickEntryPreview> preview{};
};

struct PickAssistObservation {
    uint64_t controller_tick = 0U;
    RuntimeState runtime_state = RuntimeState::Locomotion;
    const InteractionTarget* target = nullptr;
    Transform displayed_root{};
    vec3 simulation_velocity{};
    float displayed_planar_speed_mps = 0.0F;
    float camera_azimuth = 0.0F;
    uint64_t snapshot_fingerprint = 0U;
    uint64_t preview_snapshot_fingerprint = 0U;
    std::vector<PickAssistPreviewResult> preview_results{};
    std::vector<PickNavigationObstacle> live_obstacles{};
};

struct PickAssistOutput {
    bool override_steering = false;
    vec3 left_stick{};
    vec3 right_stick{};
    bool force_strafe = false;
    bool stationary_constraint = false;
    bool planning_barrier = false;
    std::vector<PickAssistPreviewRequest> preview_requests{};
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

struct PickAssistSelectionPreviewDiagnostics {
    PickAssistPreviewRequest request{};
    std::optional<PickEntryRoot> prospective_root{};
    uint64_t observation_snapshot_fingerprint = 0U;
    uint64_t preview_snapshot_fingerprint = 0U;
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
    uint32_t selected_slot_id = 0U;
    PickSlotSelection slot_selection{};
    std::optional<size_t> frozen_slot_index{};
    std::vector<PickAssistSelectionPreviewDiagnostics> selection_previews{};
    uint32_t selection_preview_epochs = 0U;
    uint32_t selection_preview_calls = 0U;
    bool selection_poor_match_observed = false;
    uint32_t settle_ticks = 0U;
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
    std::vector<PickAssistPreviewRequest> selection_preview_requests_{};
    bool selection_preview_outstanding_ = false;
    MappedPickSlot frozen_slot_{};
    std::optional<PickEntryRoot> frozen_preview_root_{};
    bool poor_match_observed_ = false;
    Transform previous_observed_root_{};
    uint32_t arrival_ticks_ = 0U;
    PickAssistDiagnostics diagnostics_{};
};

const char* pick_assist_state_name(PickAssistState state);
const char* pick_assist_reason_name(PickAssistReason reason);
PickAssistReason pick_assist_reason_from_slot_reason(PickSlotReason reason);

}  // namespace interaction
