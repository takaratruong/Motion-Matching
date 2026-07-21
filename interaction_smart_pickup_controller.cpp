#include "interaction_smart_pickup_controller.h"

#include "g1_skeleton.h"
#include "interaction_funnel_worker.h"
#include "interaction_learned_pickup_backend.h"
#include "interaction_sha256.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace interaction {
namespace {

class ProductionSmartPickupAssistBackend final
    : public SmartPickupAssistBackend {
public:
    bool begin(
        const PickAssistStart& start,
        const InteractionTarget* post_step_target) override {
        return implementation_.begin(start, post_step_target);
    }

    void cancel() override {
        implementation_.cancel();
    }

    PickAssistOutput observe(
        const PickAssistObservation& observation) override {
        return implementation_.observe(observation);
    }

    std::optional<PickRequest> take_submission(
        uint64_t request_id) override {
        return implementation_.take_submission(request_id);
    }

    bool active() const override {
        return implementation_.active();
    }

    bool owns_manual_interact() const override {
        return implementation_.owns_manual_interact();
    }

    const PickAssistDiagnostics& diagnostics() const override {
        return implementation_.diagnostics();
    }

private:
    ControllerPickAssist implementation_{};
};

Transform snapshot_root(const LocomotionSnapshot& snapshot) {
    return {
        snapshot.pose.positions[g1_skeleton::Simulation],
        snapshot.pose.rotations[g1_skeleton::Simulation],
    };
}

std::optional<PickEntryPreview> preview_frozen_smart_pickup(
    const SmartPickupPreviewCallback& preview_pick,
    const LocomotionSnapshot& live_snapshot,
    PickEntryRoot frozen_root,
    TargetHandle target,
    uint32_t affordance_id) {
    return preview_pick(live_snapshot, frozen_root, target, affordance_id);
}

std::vector<PickNavigationObstacle> collect_live_obstacles(
    const SmartPickupPostStepInput& input) {
    if (input.obstacle_centers.size() != input.obstacle_sizes.size()) {
        throw std::invalid_argument(
            "smart-pickup obstacle arrays must have equal length");
    }
    if (!input.live_obstacles.empty()) return input.live_obstacles;
    std::vector<PickNavigationObstacle> obstacles;
    obstacles.reserve(input.obstacle_centers.size());
    for (size_t index = 0U; index < input.obstacle_centers.size(); ++index) {
        obstacles.push_back({
            input.obstacle_centers[index], input.obstacle_sizes[index]});
    }
    return obstacles;
}

}  // namespace

SmartPickupProductionConfig parse_smart_pickup_production_config(
    const std::map<std::string, std::string>& environment) {
    SmartPickupProductionConfig config{};
    const auto provider = environment.find("G1_SMART_PICKUP_PROVIDER");
    if (provider == environment.end()) return config;
    if (provider->second == "authored") return config;
    if (provider->second != "learned") {
        throw std::invalid_argument(
            "G1_SMART_PICKUP_PROVIDER must be authored or learned");
    }

    const auto required_path = [&environment](const char* name) {
        const auto value = environment.find(name);
        if (value == environment.end() || value->second.empty()) {
            throw std::invalid_argument(
                std::string("learned Smart Pickup requires ") + name);
        }
        return std::filesystem::path(value->second);
    };
    config.mode = SmartPickupProviderMode::Learned;
    config.python = required_path("G1_FUNNEL_PYTHON");
    config.checkpoint = required_path("G1_FUNNEL_CHECKPOINT");
    config.worker = required_path("G1_FUNNEL_WORKER");
    config.work_directory = required_path("G1_FUNNEL_WORK_DIR");
    for (const auto& path :
         {config.python, config.checkpoint, config.worker}) {
        if (!std::filesystem::is_regular_file(path)) {
            throw std::invalid_argument(
                "learned Smart Pickup file is unavailable: " + path.string());
        }
    }
    std::filesystem::create_directories(config.work_directory);
    config.checkpoint_sha256 = sha256_file_bytes(config.checkpoint);
    return config;
}

SmartPickupProductionConfig load_smart_pickup_production_config() {
    std::map<std::string, std::string> environment;
    const char* names[]{
        "G1_SMART_PICKUP_PROVIDER",
        "G1_FUNNEL_PYTHON",
        "G1_FUNNEL_CHECKPOINT",
        "G1_FUNNEL_WORKER",
        "G1_FUNNEL_WORK_DIR",
    };
    for (const char* name : names) {
        if (const char* value = std::getenv(name); value != nullptr) {
            environment.emplace(name, value);
        }
    }
    return parse_smart_pickup_production_config(environment);
}

SmartPickupController::SmartPickupController()
    : SmartPickupController(SmartPickupProductionConfig{}) {}

SmartPickupController::SmartPickupController(
    const SmartPickupProductionConfig& config) {
    if (config.mode == SmartPickupProviderMode::Authored) {
        owned_backend_ =
            std::make_unique<ProductionSmartPickupAssistBackend>();
    } else {
        owned_provider_ = std::make_unique<AsyncPythonFunnelProvider>(
            config.python,
            config.worker,
            config.checkpoint,
            config.work_directory);
        LearnedPickupConfig learned{};
        learned.checkpoint_sha256 = config.checkpoint_sha256;
        owned_backend_ = std::make_unique<LearnedSmartPickupBackend>(
            *owned_provider_, learned);
    }
    backend_ = owned_backend_.get();
}

SmartPickupController::SmartPickupController(
    SmartPickupAssistBackend& backend)
    : backend_(&backend) {}

SmartPickupController::~SmartPickupController() = default;

SmartPickupPreStepResult SmartPickupController::pre_step(
    const SmartPickupPreStepInput& input) {
    SmartPickupPreStepResult result{};
    result.left_stick = input.left_stick;
    result.right_stick = input.right_stick;
    result.force_strafe = input.force_strafe;

    const bool owns_cancel =
        pending_activation_.has_value() ||
        backend_->active() ||
        backend_->owns_manual_interact() ||
        (input.interact_pressed &&
         input.runtime_state == RuntimeState::Locomotion);
    if (input.cancel_pressed && owns_cancel) {
        pending_activation_.reset();
        previous_assist_output_ = {};
        backend_->cancel();
        result.cancel_consumed = true;
        result.interact_consumed = input.interact_pressed;
        return result;
    }

    const bool owns_existing_attempt =
        pending_activation_.has_value() ||
        backend_->active() ||
        backend_->owns_manual_interact();
    if (input.manual_override_pressed && owns_existing_attempt) {
        pending_activation_.reset();
        previous_assist_output_ = {};
        backend_->cancel();
        result.manual_override_consumed = true;
        result.interact_consumed = input.interact_pressed;
        return result;
    }

    if (pending_activation_.has_value()) {
        result.interact_consumed = input.interact_pressed;
        result.left_stick = {};
        result.right_stick = {};
        result.force_strafe = false;
        return result;
    }

    if (backend_->active()) {
        result.interact_consumed =
            input.interact_pressed && backend_->owns_manual_interact();
        if (previous_assist_output_.override_steering) {
            result.left_stick = previous_assist_output_.left_stick;
            result.right_stick = previous_assist_output_.right_stick;
            result.force_strafe = previous_assist_output_.force_strafe;
        }
        return result;
    }

    if (input.interact_pressed &&
        input.runtime_state == RuntimeState::Locomotion) {
        PendingManualPickActivation pending{};
        if (input.selected_target != nullptr) {
            pending.target_snapshot = *input.selected_target;
        }
        pending.affordance_id =
            input.selected_affordance_id.value_or(0U);
        pending_activation_ = std::move(pending);

        result.interact_consumed = true;
        result.left_stick = {};
        result.right_stick = {};
        result.force_strafe = false;
    }
    return result;
}

SmartPickupPostStepResult SmartPickupController::post_step(
    const SmartPickupPostStepInput& input,
    const SmartPickupPreviewCallback& preview_pick) {
    SmartPickupPostStepResult result{};
    result.snapshot_fingerprint = runtime_detail::
        locomotion_snapshot_fingerprint(input.live_flat_snapshot);

    bool observe_assist = false;
    std::vector<PickNavigationObstacle> live_obstacles{};
    if (pending_activation_.has_value()) {
        std::optional<PendingManualPickActivation> activation =
            std::exchange(pending_activation_, std::nullopt);
        live_obstacles = collect_live_obstacles(input);
        PickAssistStart start{};
        start.target_snapshot = activation->target_snapshot;
        start.affordance_id = activation->affordance_id;
        start.root_world = snapshot_root(input.live_flat_snapshot);
        start.obstacles = live_obstacles;
        start.controller_tick = input.controller_tick;
        start.live_obstacles = live_obstacles;

        backend_->begin(start, input.current_target);
        observe_assist = true;
    } else {
        observe_assist = backend_->active();
        if (observe_assist) live_obstacles = collect_live_obstacles(input);
    }

    if (!observe_assist) {
        previous_assist_output_ = {};
        return result;
    }

    PickAssistObservation observation{};
    observation.controller_tick = input.controller_tick;
    observation.runtime_state = input.runtime_state;
    observation.target = input.current_target;
    observation.displayed_root = snapshot_root(input.live_flat_snapshot);
    observation.simulation_velocity = input.simulation_velocity;
    observation.displayed_planar_speed_mps =
        input.displayed_planar_speed_mps;
    observation.camera_azimuth = input.camera_azimuth;
    observation.snapshot_fingerprint = result.snapshot_fingerprint;
    observation.live_obstacles = live_obstacles;

    if (!previous_assist_output_.preview_requests.empty() &&
        static_cast<bool>(preview_pick)) {
        const PickAssistDiagnostics& frozen = backend_->diagnostics();
        observation.preview_results.reserve(
            previous_assist_output_.preview_requests.size());
        for (const PickAssistPreviewRequest& request :
             previous_assist_output_.preview_requests) {
            observation.preview_results.push_back({
                request,
                preview_frozen_smart_pickup(
                    preview_pick,
                    input.live_flat_snapshot,
                    request.root,
                    frozen.target,
                    frozen.affordance_id),
            });
        }
        observation.preview_snapshot_fingerprint =
            result.snapshot_fingerprint;
    }

    result.assist_output = backend_->observe(observation);
    if (result.assist_output.submit_interact) {
        result.pick_request = backend_->take_submission(
            input.next_request_id);
    }
    previous_assist_output_ = result.assist_output;
    return result;
}

const PickAssistDiagnostics& SmartPickupController::diagnostics() const {
    return backend_->diagnostics();
}

std::optional<LearnedPickupDebugSnapshot>
SmartPickupController::learned_debug_snapshot() const {
    return backend_->learned_debug_snapshot();
}

}  // namespace interaction
