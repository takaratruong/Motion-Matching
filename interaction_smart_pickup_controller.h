#pragma once

#include "interaction_funnel_timing.h"
#include "interaction_pick_assist.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace interaction {

class FunnelProposalProvider;

enum class SmartPickupProviderMode {
    Authored,
    Learned,
};

struct SmartPickupProductionConfig {
    SmartPickupProviderMode mode = SmartPickupProviderMode::Authored;
    std::filesystem::path python{};
    std::filesystem::path checkpoint{};
    std::filesystem::path worker{};
    std::filesystem::path work_directory{};
    std::array<uint8_t, 32> checkpoint_sha256{};
};

SmartPickupProductionConfig parse_smart_pickup_production_config(
    const std::map<std::string, std::string>& environment);
SmartPickupProductionConfig load_smart_pickup_production_config();

using SmartPickupPreviewCallback = std::function<std::optional<PickEntryPreview>(
    const LocomotionSnapshot&,
    PickEntryRoot,
    TargetHandle,
    uint32_t)>;

struct LearnedPickupDebugSnapshot {
    bool active = false;
    FunnelSample frozen_entry_world{};
    FunnelSample tracked_root_world{};
    FunnelSample terminal_root_world{};
    std::vector<FunnelSample> world_route{};
    int progress_index = 0;
    int lookahead_index = 0;
};

class SmartPickupAssistBackend {
public:
    virtual ~SmartPickupAssistBackend() = default;

    virtual bool begin(
        const PickAssistStart& start,
        const InteractionTarget* post_step_target) = 0;
    virtual void cancel() = 0;
    virtual PickAssistOutput observe(
        const PickAssistObservation& observation) = 0;
    virtual std::optional<PickRequest> take_submission(
        uint64_t request_id) = 0;
    virtual std::optional<PickEntryPreview> take_certified_preview() {
        return std::nullopt;
    }
    virtual bool active() const = 0;
    virtual bool owns_manual_interact() const = 0;
    virtual const PickAssistDiagnostics& diagnostics() const = 0;
    virtual std::optional<LearnedPickupDebugSnapshot>
    learned_debug_snapshot() const {
        return std::nullopt;
    }
};

struct SmartPickupPreStepInput {
    RuntimeState runtime_state = RuntimeState::Locomotion;
    bool interact_pressed = false;
    bool cancel_pressed = false;
    bool manual_override_pressed = false;
    const InteractionTarget* selected_target = nullptr;
    std::optional<uint32_t> selected_affordance_id{};
    vec3 left_stick{};
    vec3 right_stick{};
    bool force_strafe = false;
};

struct SmartPickupPreStepResult {
    bool interact_consumed = false;
    bool cancel_consumed = false;
    bool manual_override_consumed = false;
    vec3 left_stick{};
    vec3 right_stick{};
    bool force_strafe = false;
};

struct SmartPickupPostStepInput {
    uint64_t controller_tick = 0U;
    RuntimeState runtime_state = RuntimeState::Locomotion;
    LocomotionSnapshot live_flat_snapshot{};
    const InteractionTarget* current_target = nullptr;
    std::vector<vec3> obstacle_centers{};
    std::vector<vec3> obstacle_sizes{};
    std::vector<PickNavigationObstacle> live_obstacles{};
    vec3 simulation_velocity{};
    float displayed_planar_speed_mps = 0.0F;
    float camera_azimuth = 0.0F;
    uint64_t next_request_id = 0U;
};

struct SmartPickupPostStepResult {
    PickAssistOutput assist_output{};
    std::optional<PickRequest> pick_request{};
    std::optional<PickEntryPreview> certified_preview{};
    uint64_t snapshot_fingerprint = 0U;
};

class SmartPickupController {
public:
    SmartPickupController();
    explicit SmartPickupController(const SmartPickupProductionConfig& config);
    explicit SmartPickupController(SmartPickupAssistBackend& backend);
    ~SmartPickupController();

    SmartPickupController(const SmartPickupController&) = delete;
    SmartPickupController& operator=(const SmartPickupController&) = delete;
    SmartPickupController(SmartPickupController&&) = delete;
    SmartPickupController& operator=(SmartPickupController&&) = delete;

    SmartPickupPreStepResult pre_step(
        const SmartPickupPreStepInput& input);
    SmartPickupPostStepResult post_step(
        const SmartPickupPostStepInput& input,
        const SmartPickupPreviewCallback& preview_pick);

    const PickAssistDiagnostics& diagnostics() const;
    std::optional<LearnedPickupDebugSnapshot>
    learned_debug_snapshot() const;

private:
    struct PendingManualPickActivation {
        InteractionTarget target_snapshot{};
        uint32_t affordance_id = 0U;
    };

    std::unique_ptr<FunnelProposalProvider> owned_provider_{};
    std::unique_ptr<SmartPickupAssistBackend> owned_backend_{};
    SmartPickupAssistBackend* backend_ = nullptr;
    std::optional<PendingManualPickActivation> pending_activation_{};
    PickAssistOutput previous_assist_output_{};
};

}  // namespace interaction
