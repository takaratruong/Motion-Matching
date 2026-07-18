#include "interaction_smart_pickup_controller.h"

#include "g1_skeleton.h"
#include "interaction_pick_slots.h"
#include "interaction_smart_pickup_scene.h"
#include "locomotion_timing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using ExpectedPreviewCallback = std::function<std::optional<
    interaction::PickEntryPreview>(
        const interaction::LocomotionSnapshot&,
        interaction::PickEntryRoot,
        interaction::TargetHandle,
        uint32_t)>;

static_assert(std::is_same_v<
    interaction::SmartPickupPreviewCallback,
    ExpectedPreviewCallback>);
static_assert(std::is_same_v<
    decltype(std::declval<interaction::SmartPickupController&>().pre_step(
        std::declval<const interaction::SmartPickupPreStepInput&>())),
    interaction::SmartPickupPreStepResult>);
static_assert(std::is_same_v<
    decltype(std::declval<interaction::SmartPickupController&>().post_step(
        std::declval<const interaction::SmartPickupPostStepInput&>(),
        std::declval<const interaction::SmartPickupPreviewCallback&>())),
    interaction::SmartPickupPostStepResult>);
static_assert(std::has_virtual_destructor_v<
    interaction::SmartPickupAssistBackend>);
static_assert(std::is_default_constructible_v<
    interaction::SmartPickupController>);
static_assert(std::is_constructible_v<
    interaction::SmartPickupController,
    interaction::SmartPickupAssistBackend&>);
static_assert(locomotion_timing::kRateHz == 25);
static_assert(locomotion_timing::kStepSeconds == 1.0F / 25.0F);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool same_float_bits(float left, float right) {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

bool same_vec3_bits(vec3 left, vec3 right) {
    return same_float_bits(left.x, right.x) &&
        same_float_bits(left.y, right.y) &&
        same_float_bits(left.z, right.z);
}

bool same_quat_bits(quat left, quat right) {
    return same_float_bits(left.w, right.w) &&
        same_float_bits(left.x, right.x) &&
        same_float_bits(left.y, right.y) &&
        same_float_bits(left.z, right.z);
}

bool same_transform_bits(
    interaction::Transform left,
    interaction::Transform right) {
    return same_vec3_bits(left.position, right.position) &&
        same_quat_bits(left.rotation, right.rotation);
}

bool same_pick_entry_root_bits(
    interaction::PickEntryRoot left,
    interaction::PickEntryRoot right) {
    return same_float_bits(left.world_x, right.world_x) &&
        same_float_bits(left.world_z, right.world_z) &&
        same_float_bits(
            left.world_yaw_radians, right.world_yaw_radians);
}

bool same_snapshot_bits(
    const interaction::LocomotionSnapshot& left,
    const interaction::LocomotionSnapshot& right) {
    for (size_t bone = 0U; bone < left.pose.positions.size(); ++bone) {
        if (!same_vec3_bits(
                left.pose.positions[bone], right.pose.positions[bone]) ||
            !same_vec3_bits(
                left.pose.velocities[bone], right.pose.velocities[bone]) ||
            !same_quat_bits(
                left.pose.rotations[bone], right.pose.rotations[bone]) ||
            !same_vec3_bits(
                left.pose.angular_velocities[bone],
                right.pose.angular_velocities[bone])) {
            return false;
        }
    }
    for (size_t index = 0U; index < left.pose.hand_dof.size(); ++index) {
        if (!same_float_bits(
                left.pose.hand_dof[index], right.pose.hand_dof[index]) ||
            !same_float_bits(
                left.pose.hand_dof_velocities[index],
                right.pose.hand_dof_velocities[index])) {
            return false;
        }
    }
    if (left.pose.foot_contacts != right.pose.foot_contacts) return false;
    for (size_t index = 0U;
         index < left.future_root_positions.size();
         ++index) {
        if (!same_vec3_bits(
                left.future_root_positions[index],
                right.future_root_positions[index]) ||
            !same_quat_bits(
                left.future_root_rotations[index],
                right.future_root_rotations[index])) {
            return false;
        }
    }
    return true;
}

bool same_vec3_vector_bits(
    const std::vector<vec3>& left,
    const std::vector<vec3>& right) {
    if (left.size() != right.size()) return false;
    for (size_t index = 0U; index < left.size(); ++index) {
        if (!same_vec3_bits(left[index], right[index])) return false;
    }
    return true;
}

bool same_navigation_obstacle_bits(
    const std::vector<interaction::PickNavigationObstacle>& obstacles,
    const std::vector<vec3>& centers,
    const std::vector<vec3>& sizes) {
    if (obstacles.size() != centers.size() ||
        obstacles.size() != sizes.size()) {
        return false;
    }
    for (size_t index = 0U; index < obstacles.size(); ++index) {
        if (!same_vec3_bits(
                obstacles[index].center_world, centers[index]) ||
            !same_vec3_bits(
                obstacles[index].size_world, sizes[index])) {
            return false;
        }
    }
    return true;
}

bool is_zero(vec3 value) {
    return value.x == 0.0F && value.y == 0.0F && value.z == 0.0F;
}

interaction::Transform root_transform(
    const interaction::LocomotionSnapshot& snapshot) {
    return {
        snapshot.pose.positions[g1_skeleton::Simulation],
        snapshot.pose.rotations[g1_skeleton::Simulation],
    };
}

interaction::InteractionTarget make_controller_target() {
    interaction::InteractionTarget target =
        interaction::make_smart_pickup_demo_target();
    target.handle = {71U, 9U};
    target.object_world = {
        vec3(0.0F, 0.80F, 0.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    target.table_world = {
        vec3(100.0F, 0.0F, 100.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    target.table_size = vec3(1.0F, 0.10F, 1.0F);
    target.state = interaction::ObjectState::Free;
    target.owner_request = 0U;
    require(
        target.affordances.size() == 1U,
        "smart-pickup fixture did not have one affordance");
    interaction::GraspAffordance& affordance = target.affordances.front();
    affordance.id = 7U;
    affordance.interaction_slots = {
        {11U, 0.0F, 0.0F, 0.0F},
        {22U, -0.80F, -0.20F, -0.70F},
        {33U, 0.80F, -0.20F, 0.70F},
    };
    return target;
}

interaction::LocomotionSnapshot make_snapshot(
    vec3 root_position,
    float root_yaw_radians) {
    interaction::LocomotionSnapshot snapshot{};
    snapshot.pose.rotations.fill(quat(1.0F, 0.0F, 0.0F, 0.0F));
    snapshot.pose.positions[g1_skeleton::Simulation] = root_position;
    snapshot.pose.rotations[g1_skeleton::Simulation] =
        quat_from_angle_axis(
            root_yaw_radians, vec3(0.0F, 1.0F, 0.0F));
    for (size_t index = 0U;
         index < snapshot.future_root_positions.size();
         ++index) {
        const float scale = static_cast<float>(index + 1U);
        snapshot.future_root_positions[index] =
            root_position + vec3(0.01F * scale, 0.0F, -0.02F * scale);
        snapshot.future_root_rotations[index] =
            snapshot.pose.rotations[g1_skeleton::Simulation];
    }
    return snapshot;
}

class CountingAssistBackend final
    : public interaction::SmartPickupAssistBackend {
public:
    explicit CountingAssistBackend(
        interaction::PickAssistConfig config = {})
        : implementation_(config) {}

    bool begin(
        const interaction::PickAssistStart& start,
        const interaction::InteractionTarget* post_step_target) override {
        ++begin_calls;
        begin_inputs.push_back(start);
        begin_had_post_step_target.push_back(post_step_target != nullptr);
        if (post_step_target != nullptr) {
            begin_post_step_targets.push_back(*post_step_target);
        } else {
            begin_post_step_targets.emplace_back();
        }
        return implementation_.begin(start, post_step_target);
    }

    void cancel() override {
        ++cancel_calls;
        implementation_.cancel();
    }

    interaction::PickAssistOutput observe(
        const interaction::PickAssistObservation& observation) override {
        ++observe_calls;
        observations.push_back(observation);
        return implementation_.observe(observation);
    }

    std::optional<interaction::PickRequest> take_submission(
        uint64_t request_id) override {
        ++take_submission_calls;
        submission_request_ids.push_back(request_id);
        std::optional<interaction::PickRequest> request =
            implementation_.take_submission(request_id);
        if (request.has_value()) ++yielded_submissions;
        return request;
    }

    bool active() const override {
        return implementation_.active();
    }

    bool owns_manual_interact() const override {
        return implementation_.owns_manual_interact();
    }

    const interaction::PickAssistDiagnostics& diagnostics()
        const override {
        return implementation_.diagnostics();
    }

    uint32_t begin_calls = 0U;
    uint32_t cancel_calls = 0U;
    uint32_t observe_calls = 0U;
    uint32_t take_submission_calls = 0U;
    uint32_t yielded_submissions = 0U;
    std::vector<interaction::PickAssistStart> begin_inputs{};
    std::vector<bool> begin_had_post_step_target{};
    std::vector<interaction::InteractionTarget> begin_post_step_targets{};
    std::vector<interaction::PickAssistObservation> observations{};
    std::vector<uint64_t> submission_request_ids{};

private:
    interaction::ControllerPickAssist implementation_;
};

class ThrowingBeginAssistBackend final
    : public interaction::SmartPickupAssistBackend {
public:
    bool begin(
        const interaction::PickAssistStart&,
        const interaction::InteractionTarget*) override {
        ++begin_calls;
        throw std::runtime_error("intentional smart-pickup begin failure");
    }

    void cancel() override {
        ++cancel_calls;
    }

    interaction::PickAssistOutput observe(
        const interaction::PickAssistObservation&) override {
        ++observe_calls;
        return {};
    }

    std::optional<interaction::PickRequest> take_submission(
        uint64_t) override {
        ++take_submission_calls;
        return std::nullopt;
    }

    bool active() const override {
        return false;
    }

    bool owns_manual_interact() const override {
        return false;
    }

    const interaction::PickAssistDiagnostics& diagnostics()
        const override {
        return diagnostics_;
    }

    uint32_t begin_calls = 0U;
    uint32_t cancel_calls = 0U;
    uint32_t observe_calls = 0U;
    uint32_t take_submission_calls = 0U;

private:
    interaction::PickAssistDiagnostics diagnostics_{};
};

class InactiveStateAssistBackend final
    : public interaction::SmartPickupAssistBackend {
public:
    explicit InactiveStateAssistBackend(
        interaction::PickAssistState state) {
        diagnostics_.state = state;
    }

    bool begin(
        const interaction::PickAssistStart&,
        const interaction::InteractionTarget*) override {
        ++begin_calls;
        return false;
    }

    void cancel() override {
        ++cancel_calls;
    }

    interaction::PickAssistOutput observe(
        const interaction::PickAssistObservation&) override {
        ++observe_calls;
        return {};
    }

    std::optional<interaction::PickRequest> take_submission(
        uint64_t) override {
        ++take_submission_calls;
        return std::nullopt;
    }

    bool active() const override {
        return false;
    }

    bool owns_manual_interact() const override {
        return false;
    }

    const interaction::PickAssistDiagnostics& diagnostics()
        const override {
        return diagnostics_;
    }

    uint32_t begin_calls = 0U;
    uint32_t cancel_calls = 0U;
    uint32_t observe_calls = 0U;
    uint32_t take_submission_calls = 0U;

private:
    interaction::PickAssistDiagnostics diagnostics_{};
};

// This raylib-free harness establishes only the caller-side pre/ordinary/post
// order around the coordinator. Production locomotion-provider, live-flat
// bridge, scheduler-publication, and controller-obstacle identity counts remain
// obligations of the migrated live-flat oracle before GREEN.
struct CallerOrderHarness {
    vec3 root_position{0.0F, 0.0F, -0.80F};
    float root_yaw_radians = 0.0F;
    vec3 ordinary_displacement{};
    float stick_displacement_scale = 0.04F;
    uint32_t ordinary_step_calls = 0U;
    uint32_t materialized_snapshot_count = 0U;
    uint32_t caller_publication_count = 0U;
    interaction::LocomotionSnapshot last_snapshot{};

    interaction::LocomotionSnapshot step(
        const interaction::SmartPickupPreStepResult& input) {
        ++ordinary_step_calls;
        root_position = root_position + ordinary_displacement +
            stick_displacement_scale * input.left_stick;
        ++materialized_snapshot_count;
        last_snapshot = make_snapshot(root_position, root_yaw_radians);
        return last_snapshot;
    }

    uint64_t publish_same_snapshot(
        const interaction::LocomotionSnapshot& snapshot) {
        ++caller_publication_count;
        require(
            same_snapshot_bits(snapshot, last_snapshot),
            "caller did not publish its materialized snapshot unchanged");
        return interaction::runtime_detail::
            locomotion_snapshot_fingerprint(snapshot);
    }
};

interaction::SmartPickupPreStepInput make_pre_input(
    const interaction::InteractionTarget* target,
    bool interact_pressed,
    bool cancel_pressed = false) {
    interaction::SmartPickupPreStepInput input{};
    input.runtime_state = interaction::RuntimeState::Locomotion;
    input.interact_pressed = interact_pressed;
    input.cancel_pressed = cancel_pressed;
    input.selected_target = target;
    if (target != nullptr && !target->affordances.empty()) {
        input.selected_affordance_id = target->affordances.front().id;
    }
    input.left_stick = vec3(0.75F, 0.0F, -0.25F);
    input.right_stick = vec3(-0.30F, 0.0F, 0.90F);
    input.force_strafe = false;
    return input;
}

interaction::SmartPickupPostStepInput make_post_input(
    const interaction::LocomotionSnapshot& snapshot,
    const interaction::InteractionTarget* target,
    const std::vector<vec3>& obstacle_centers = {},
    const std::vector<vec3>& obstacle_sizes = {}) {
    interaction::SmartPickupPostStepInput input{};
    input.runtime_state = interaction::RuntimeState::Locomotion;
    input.live_flat_snapshot = snapshot;
    input.current_target = target;
    input.obstacle_centers = obstacle_centers;
    input.obstacle_sizes = obstacle_sizes;
    input.simulation_velocity = vec3();
    input.displayed_planar_speed_mps = 0.0F;
    input.camera_azimuth = 0.0F;
    input.next_request_id = 71U;
    return input;
}

interaction::SmartPickupPreviewCallback rejecting_preview_counter(
    uint32_t& calls) {
    return [&calls](
               const interaction::LocomotionSnapshot&,
               interaction::PickEntryRoot,
               interaction::TargetHandle,
               uint32_t) -> std::optional<interaction::PickEntryPreview> {
        ++calls;
        return std::nullopt;
    };
}

void test_activation_brackets_one_caller_step_and_defers_assist_motion() {
    interaction::InteractionTarget target = make_controller_target();
    const interaction::InteractionTarget target_before = target;
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;
    locomotion.ordinary_displacement = vec3(0.013F, 0.0F, 0.017F);
    const vec3 root_before = locomotion.root_position;

    const interaction::SmartPickupPreStepResult pre = controller.pre_step(
        make_pre_input(&target, true));
    require(pre.interact_consumed,
        "F activation did not consume the interact edge");
    require(!pre.cancel_consumed,
        "F-only activation spuriously consumed cancel");
    require(
        backend.begin_calls == 0U && backend.observe_calls == 0U,
        "pre_step began or observed assist before ordinary locomotion");
    require(is_zero(pre.left_stick) && is_zero(pre.right_stick),
        "F activation did not zero both manual sticks");
    require(!pre.force_strafe,
        "F activation synthesized strafe before locomotion");
    require(
        interaction::same_interaction_target_snapshot(target, target_before),
        "pre_step mutated the selected target");

    const interaction::LocomotionSnapshot snapshot = locomotion.step(pre);
    const interaction::LocomotionSnapshot snapshot_before = snapshot;
    const vec3 expected_ordinary_root =
        root_before + locomotion.ordinary_displacement;
    require(
        same_vec3_bits(
            root_transform(snapshot).position, expected_ordinary_root),
        "activation tick contained non-ordinary assisted displacement");
    require(
        locomotion.ordinary_step_calls == 1U &&
            locomotion.materialized_snapshot_count == 1U,
        "activation was not bracketed around one caller step and snapshot");

    uint32_t preview_calls = 0U;
    const interaction::SmartPickupPostStepResult post =
        controller.post_step(
            make_post_input(snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
            backend.take_submission_calls == 0U,
        "post_step did not begin and observe the pending assist exactly once");
    require(preview_calls == 0U && !post.pick_request.has_value(),
        "activation tick previewed or submitted a pick");
    require(
        post.assist_output.override_steering &&
            !post.assist_output.needs_preview &&
            !post.assist_output.submit_interact,
        "activation observation did not publish next-tick approach output");
    require(
        same_snapshot_bits(snapshot, snapshot_before),
        "post_step mutated the authoritative live-flat snapshot");
    require(
        interaction::same_interaction_target_snapshot(target, target_before),
        "post_step mutated the selected target");
    require(
        backend.begin_inputs.size() == 1U &&
            same_transform_bits(
                backend.begin_inputs.front().root_world,
                root_transform(snapshot)) &&
            backend.begin_inputs.front().obstacles.empty() &&
            interaction::same_interaction_target_snapshot(
                backend.begin_inputs.front().target_snapshot,
                target_before) &&
            backend.begin_had_post_step_target.front() &&
            interaction::same_interaction_target_snapshot(
                backend.begin_post_step_targets.front(), target),
        "backend begin did not receive exact pre-target and post-root inputs");
    require(
        backend.observations.size() == 1U &&
            same_transform_bits(
                backend.observations.front().displayed_root,
                root_transform(snapshot)) &&
            backend.observations.front().snapshot_fingerprint ==
                post.snapshot_fingerprint,
        "backend observation did not use the authoritative post-step snapshot");

    const interaction::PickSlotSelection expected_selection =
        interaction::select_pick_slot(
            root_transform(snapshot),
            target,
            target.affordances.front(),
            {});
    const interaction::PickAssistDiagnostics& diagnostics =
        controller.diagnostics();
    require(
        expected_selection.selected_index.has_value() &&
            diagnostics.slot_selection.selected_index.has_value(),
        "post-step root fixture did not produce a selected slot");
    const interaction::MappedPickSlot& expected_slot =
        expected_selection.ordered[*expected_selection.selected_index];
    const interaction::MappedPickSlot& actual_slot =
        diagnostics.slot_selection.ordered[
            *diagnostics.slot_selection.selected_index];
    require(
        diagnostics.state == interaction::PickAssistState::SlotApproach &&
            diagnostics.reason == interaction::PickAssistReason::None &&
            diagnostics.selected_slot_id == expected_slot.id &&
            actual_slot.id == expected_slot.id &&
            same_transform_bits(
                actual_slot.root_world, expected_slot.root_world) &&
            same_float_bits(
                diagnostics.route_length_m,
                expected_slot.route_length_m),
        "assist did not freeze its slot from the post-step root");

    const uint64_t published_fingerprint =
        locomotion.publish_same_snapshot(snapshot);
    require(
        post.snapshot_fingerprint == published_fingerprint &&
            locomotion.caller_publication_count == 1U,
        "observation and caller publication fingerprints differed");

    interaction::SmartPickupPreStepInput next_input =
        make_pre_input(&target, false);
    next_input.left_stick = vec3(-1.0F, 0.0F, -1.0F);
    next_input.right_stick = vec3(1.0F, 0.0F, 1.0F);
    const interaction::SmartPickupPreStepResult next =
        controller.pre_step(next_input);
    require(
        !next.interact_consumed && !next.cancel_consumed &&
            same_vec3_bits(
                next.left_stick, post.assist_output.left_stick) &&
            same_vec3_bits(
                next.right_stick, post.assist_output.right_stick) &&
            next.force_strafe == post.assist_output.force_strafe,
        "active assist did not own both movement sticks on the following tick");
    locomotion.ordinary_displacement = vec3();
    const vec3 next_root_before = locomotion.root_position;
    const interaction::LocomotionSnapshot next_snapshot =
        locomotion.step(next);
    require(
        !same_vec3_bits(
            root_transform(next_snapshot).position, next_root_before),
        "assisted steering did not affect the tick after activation");
}

void test_begin_exception_clears_pending_activation_before_backend_call() {
    interaction::InteractionTarget target = make_controller_target();
    ThrowingBeginAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    require(
        activation.interact_consumed && is_zero(activation.left_stick) &&
            is_zero(activation.right_stick),
        "throwing-begin fixture did not capture the F edge");
    const interaction::LocomotionSnapshot activation_snapshot =
        locomotion.step(activation);

    uint32_t preview_calls = 0U;
    bool begin_threw = false;
    try {
        (void)controller.post_step(
            make_post_input(activation_snapshot, &target),
            rejecting_preview_counter(preview_calls));
    } catch (const std::runtime_error& error) {
        begin_threw = std::string(error.what()) ==
            "intentional smart-pickup begin failure";
    }
    require(
        begin_threw && backend.begin_calls == 1U &&
            backend.observe_calls == 0U && preview_calls == 0U,
        "backend begin exception was not propagated exactly once");

    interaction::SmartPickupPreStepInput raw =
        make_pre_input(&target, false);
    raw.left_stick = vec3(-0.81F, 0.0F, 0.37F);
    raw.right_stick = vec3(0.26F, 0.0F, -0.93F);
    raw.force_strafe = true;
    const interaction::SmartPickupPreStepResult after_exception =
        controller.pre_step(raw);
    require(
        !after_exception.interact_consumed &&
            !after_exception.cancel_consumed &&
            same_vec3_bits(after_exception.left_stick, raw.left_stick) &&
            same_vec3_bits(after_exception.right_stick, raw.right_stick) &&
            after_exception.force_strafe == raw.force_strafe &&
            backend.begin_calls == 1U,
        "begin exception left pending activation owning the next pre-step");

    const interaction::LocomotionSnapshot after_snapshot =
        locomotion.step(after_exception);
    const interaction::SmartPickupPostStepResult after_post =
        controller.post_step(
            make_post_input(after_snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 1U && backend.observe_calls == 0U &&
            backend.take_submission_calls == 0U && preview_calls == 0U &&
            !after_post.pick_request.has_value(),
        "begin exception retried pending activation on a later post-step");
}

void test_obstacle_array_exception_clears_pending_activation() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    const interaction::LocomotionSnapshot activation_snapshot =
        locomotion.step(activation);
    interaction::SmartPickupPostStepInput mismatched =
        make_post_input(activation_snapshot, &target);
    mismatched.obstacle_centers.push_back(vec3(1.0F, 2.0F, 3.0F));

    uint32_t preview_calls = 0U;
    bool mismatch_threw = false;
    try {
        (void)controller.post_step(
            mismatched, rejecting_preview_counter(preview_calls));
    } catch (const std::invalid_argument&) {
        mismatch_threw = true;
    }
    require(
        mismatch_threw && backend.begin_calls == 0U &&
            backend.observe_calls == 0U && preview_calls == 0U,
        "obstacle-array mismatch did not fail before backend work");

    interaction::SmartPickupPreStepInput raw =
        make_pre_input(&target, false);
    raw.left_stick = vec3(-0.52F, 0.0F, 0.43F);
    raw.right_stick = vec3(0.71F, 0.0F, -0.19F);
    const interaction::SmartPickupPreStepResult after_exception =
        controller.pre_step(raw);
    require(
        same_vec3_bits(after_exception.left_stick, raw.left_stick) &&
            same_vec3_bits(after_exception.right_stick, raw.right_stick),
        "obstacle-array exception left pending activation owning input");

    const interaction::LocomotionSnapshot after_snapshot =
        locomotion.step(after_exception);
    const interaction::SmartPickupPostStepResult after_post =
        controller.post_step(
            make_post_input(after_snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 0U && backend.observe_calls == 0U &&
            !after_post.pick_request.has_value(),
        "obstacle-array exception retried on a later post-step");
}

using TargetMutation = void (*)(interaction::InteractionTarget&);

void mutate_generation(interaction::InteractionTarget& target) {
    ++target.handle.generation;
}

void mutate_object_pose(interaction::InteractionTarget& target) {
    target.object_world.position.x = std::nextafter(
        target.object_world.position.x,
        std::numeric_limits<float>::infinity());
}

void mutate_state(interaction::InteractionTarget& target) {
    target.state = interaction::ObjectState::Targeted;
}

void mutate_owner(interaction::InteractionTarget& target) {
    target.owner_request = 9901U;
}

void mutate_affordance(interaction::InteractionTarget& target) {
    interaction::GraspAffordance& affordance = target.affordances.front();
    affordance.hand_in_object.position.z = std::nextafter(
        affordance.hand_in_object.position.z,
        std::numeric_limits<float>::infinity());
}

void mutate_slot_order(interaction::InteractionTarget& target) {
    std::swap(
        target.affordances.front().interaction_slots.front(),
        target.affordances.front().interaction_slots.back());
}

void mutate_slot_value(interaction::InteractionTarget& target) {
    interaction::GraspInteractionSlot& slot =
        target.affordances.front().interaction_slots.front();
    slot.root_x_object_m = std::nextafter(
        slot.root_x_object_m,
        std::numeric_limits<float>::infinity());
}

void require_pending_mutation_fails_as_target_changed(
    const char* name,
    TargetMutation mutate) {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;
    const interaction::SmartPickupPreStepResult pre = controller.pre_step(
        make_pre_input(&target, true));
    require(
        pre.interact_consumed && backend.begin_calls == 0U &&
            backend.observe_calls == 0U,
        "target-mutation fixture performed assist work before its step");

    const interaction::LocomotionSnapshot snapshot = locomotion.step(pre);
    mutate(target);
    uint32_t preview_calls = 0U;
    const interaction::SmartPickupPostStepResult post =
        controller.post_step(
            make_post_input(snapshot, &target),
            rejecting_preview_counter(preview_calls));
    const interaction::PickAssistDiagnostics& diagnostics =
        controller.diagnostics();
    const bool passed =
        locomotion.ordinary_step_calls == 1U &&
        locomotion.materialized_snapshot_count == 1U &&
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
        backend.take_submission_calls == 0U &&
        preview_calls == 0U && !post.pick_request.has_value() &&
        !post.assist_output.needs_preview &&
        !post.assist_output.submit_interact &&
        diagnostics.state == interaction::PickAssistState::Failed &&
        diagnostics.reason == interaction::PickAssistReason::TargetChanged;
    if (!passed) throw std::runtime_error(name);
}

void test_pending_target_mutations_fail_before_slot_freeze() {
    struct Case {
        const char* message;
        TargetMutation mutate;
    };
    const std::array<Case, 7> cases{{
        {"generation mutation during locomotion was not TargetChanged",
         mutate_generation},
        {"pose mutation during locomotion was not TargetChanged",
         mutate_object_pose},
        {"state mutation during locomotion was not TargetChanged",
         mutate_state},
        {"owner mutation during locomotion was not TargetChanged",
         mutate_owner},
        {"affordance mutation during locomotion was not TargetChanged",
         mutate_affordance},
        {"slot-order mutation before freeze was not TargetChanged",
         mutate_slot_order},
        {"slot-value mutation before freeze was not TargetChanged",
         mutate_slot_value},
    }};
    for (const Case& test_case : cases) {
        require_pending_mutation_fails_as_target_changed(
            test_case.message, test_case.mutate);
    }
}

void require_post_begin_slot_mutation_fails_as_slot_changed(
    const char* name,
    TargetMutation mutate) {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;
    locomotion.stick_displacement_scale = 0.0F;
    uint32_t preview_calls = 0U;

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    const interaction::LocomotionSnapshot first_snapshot =
        locomotion.step(activation);
    const interaction::SmartPickupPostStepResult began =
        controller.post_step(
            make_post_input(first_snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
            !began.pick_request.has_value() &&
            controller.diagnostics().state ==
                interaction::PickAssistState::SlotApproach,
        "post-begin slot-mutation fixture did not activate");

    mutate(target);
    const interaction::SmartPickupPreStepResult active =
        controller.pre_step(make_pre_input(&target, false));
    const interaction::LocomotionSnapshot second_snapshot =
        locomotion.step(active);
    const interaction::SmartPickupPostStepResult changed =
        controller.post_step(
            make_post_input(second_snapshot, &target),
            rejecting_preview_counter(preview_calls));
    const interaction::PickAssistDiagnostics& diagnostics =
        controller.diagnostics();
    const bool passed =
        locomotion.ordinary_step_calls == 2U &&
        locomotion.materialized_snapshot_count == 2U &&
        backend.begin_calls == 1U && backend.observe_calls == 2U &&
        backend.take_submission_calls == 0U && preview_calls == 0U &&
        !changed.pick_request.has_value() &&
        diagnostics.state == interaction::PickAssistState::Failed &&
        diagnostics.reason == interaction::PickAssistReason::SlotChanged;
    if (!passed) throw std::runtime_error(name);
}

void test_post_begin_slot_mutations_fail_without_hopping() {
    require_post_begin_slot_mutation_fails_as_slot_changed(
        "post-begin slot-order mutation was not SlotChanged",
        mutate_slot_order);
    require_post_begin_slot_mutation_fails_as_slot_changed(
        "post-begin slot-value mutation was not SlotChanged",
        mutate_slot_value);
}

void test_missing_target_and_selector_fail_stably_after_consuming_f() {
    {
        CountingAssistBackend backend;
        interaction::SmartPickupController controller(backend);
        CallerOrderHarness locomotion;
        interaction::SmartPickupPreStepInput input =
            make_pre_input(nullptr, true);
        input.selected_affordance_id = 7U;
        const interaction::SmartPickupPreStepResult pre =
            controller.pre_step(input);
        require(
            pre.interact_consumed && is_zero(pre.left_stick) &&
                is_zero(pre.right_stick) && backend.begin_calls == 0U &&
                backend.observe_calls == 0U,
            "missing-target F edge was not consumed before one caller step");
        const interaction::LocomotionSnapshot snapshot =
            locomotion.step(pre);
        uint32_t preview_calls = 0U;
        const interaction::SmartPickupPostStepResult post =
            controller.post_step(
                make_post_input(snapshot, nullptr),
                rejecting_preview_counter(preview_calls));
        const interaction::PickAssistDiagnostics first =
            controller.diagnostics();
        const interaction::PickAssistDiagnostics second =
            controller.diagnostics();
        require(
            locomotion.ordinary_step_calls == 1U &&
                locomotion.materialized_snapshot_count == 1U &&
                backend.begin_calls == 1U &&
                backend.observe_calls == 1U &&
                backend.take_submission_calls == 0U &&
                preview_calls == 0U && !post.pick_request.has_value() &&
                first.state == interaction::PickAssistState::Failed &&
                first.reason ==
                    interaction::PickAssistReason::TargetUnavailable &&
                second.state == first.state && second.reason == first.reason,
            "missing target did not fail after exactly one caller step/post");
    }
    {
        interaction::InteractionTarget target = make_controller_target();
        CountingAssistBackend backend;
        interaction::SmartPickupController controller(backend);
        CallerOrderHarness locomotion;
        interaction::SmartPickupPreStepInput input =
            make_pre_input(&target, true);
        input.selected_affordance_id.reset();
        const interaction::SmartPickupPreStepResult pre =
            controller.pre_step(input);
        require(
            pre.interact_consumed && is_zero(pre.left_stick) &&
                is_zero(pre.right_stick) && backend.begin_calls == 0U &&
                backend.observe_calls == 0U,
            "selector-failure F edge was not consumed before one caller step");
        const interaction::LocomotionSnapshot snapshot =
            locomotion.step(pre);
        uint32_t preview_calls = 0U;
        const interaction::SmartPickupPostStepResult post =
            controller.post_step(
                make_post_input(snapshot, &target),
                rejecting_preview_counter(preview_calls));
        const interaction::PickAssistDiagnostics first =
            controller.diagnostics();
        const interaction::PickAssistDiagnostics second =
            controller.diagnostics();
        require(
            locomotion.ordinary_step_calls == 1U &&
                locomotion.materialized_snapshot_count == 1U &&
                backend.begin_calls == 1U &&
                backend.observe_calls == 1U &&
                backend.take_submission_calls == 0U &&
                preview_calls == 0U && !post.pick_request.has_value() &&
                first.state == interaction::PickAssistState::Failed &&
                first.reason ==
                    interaction::PickAssistReason::TargetChanged &&
                second.state == first.state && second.reason == first.reason,
            "selector failure did not fail after exactly one caller step/post");
    }
}

void test_target_disappearing_during_step_is_observed_once_and_fails() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;
    const interaction::SmartPickupPreStepResult pre =
        controller.pre_step(make_pre_input(&target, true));
    const interaction::LocomotionSnapshot snapshot = locomotion.step(pre);
    uint32_t preview_calls = 0U;
    const interaction::SmartPickupPostStepResult post =
        controller.post_step(
            make_post_input(snapshot, nullptr),
            rejecting_preview_counter(preview_calls));
    require(
        locomotion.ordinary_step_calls == 1U &&
            locomotion.materialized_snapshot_count == 1U &&
            backend.begin_calls == 1U &&
            backend.observe_calls == 1U &&
            backend.take_submission_calls == 0U &&
            preview_calls == 0U && !post.pick_request.has_value() &&
            controller.diagnostics().state ==
                interaction::PickAssistState::Failed &&
            controller.diagnostics().reason ==
                interaction::PickAssistReason::TargetUnavailable,
        "target disappearance did not fail once after the ordinary step");
}

void test_cancel_passes_through_without_smart_pickup_ownership() {
    interaction::InteractionTarget target = make_controller_target();
    {
        InactiveStateAssistBackend backend(
            interaction::PickAssistState::Submitted);
        interaction::SmartPickupController controller(backend);
        interaction::SmartPickupPreStepInput input =
            make_pre_input(&target, false, true);
        input.runtime_state = interaction::RuntimeState::Carry;
        input.left_stick = vec3(-0.63F, 0.0F, 0.27F);
        input.right_stick = vec3(0.48F, 0.0F, -0.92F);
        input.force_strafe = true;

        const interaction::SmartPickupPreStepResult result =
            controller.pre_step(input);
        require(
            !result.cancel_consumed && !result.interact_consumed &&
                same_vec3_bits(result.left_stick, input.left_stick) &&
                same_vec3_bits(result.right_stick, input.right_stick) &&
                result.force_strafe == input.force_strafe &&
                backend.cancel_calls == 0U &&
                controller.diagnostics().state ==
                    interaction::PickAssistState::Submitted,
            "Carry X was consumed by an inactive Submitted pickup assist");
    }
    {
        CountingAssistBackend backend;
        interaction::SmartPickupController controller(backend);
        interaction::SmartPickupPreStepInput input =
            make_pre_input(&target, false, true);
        input.left_stick = vec3(0.31F, 0.0F, -0.74F);
        input.right_stick = vec3(-0.56F, 0.0F, 0.18F);
        input.force_strafe = true;

        const interaction::SmartPickupPreStepResult result =
            controller.pre_step(input);
        require(
            !result.cancel_consumed && !result.interact_consumed &&
                same_vec3_bits(result.left_stick, input.left_stick) &&
                same_vec3_bits(result.right_stick, input.right_stick) &&
                result.force_strafe == input.force_strafe &&
                backend.cancel_calls == 0U &&
                controller.diagnostics().state ==
                    interaction::PickAssistState::Idle &&
                controller.diagnostics().reason ==
                    interaction::PickAssistReason::None,
            "idle Locomotion X was consumed without a pickup attempt");
    }
    {
        InactiveStateAssistBackend backend(
            interaction::PickAssistState::Failed);
        interaction::SmartPickupController controller(backend);
        const interaction::SmartPickupPreStepInput input =
            make_pre_input(&target, false, true);
        const interaction::SmartPickupPreStepResult result =
            controller.pre_step(input);
        require(
            !result.cancel_consumed && !result.interact_consumed &&
                same_vec3_bits(result.left_stick, input.left_stick) &&
                same_vec3_bits(result.right_stick, input.right_stick) &&
                result.force_strafe == input.force_strafe &&
                backend.cancel_calls == 0U &&
                controller.diagnostics().state ==
                    interaction::PickAssistState::Failed,
            "failed inactive pickup assist consumed a later X edge");
    }
}

void test_pending_activation_owns_cancel_edge() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    require(
        activation.interact_consumed && backend.begin_calls == 0U,
        "pending-cancel fixture did not capture its activation");

    interaction::SmartPickupPreStepInput cancel =
        make_pre_input(&target, false, true);
    cancel.left_stick = vec3(-0.14F, 0.0F, 0.82F);
    cancel.right_stick = vec3(0.91F, 0.0F, -0.36F);
    const interaction::SmartPickupPreStepResult cancelled =
        controller.pre_step(cancel);
    require(
        cancelled.cancel_consumed && !cancelled.interact_consumed &&
            same_vec3_bits(cancelled.left_stick, cancel.left_stick) &&
            same_vec3_bits(cancelled.right_stick, cancel.right_stick) &&
            backend.cancel_calls == 1U && backend.begin_calls == 0U,
        "pending Smart Pickup did not own and clear X");

    CallerOrderHarness locomotion;
    const interaction::LocomotionSnapshot snapshot =
        locomotion.step(cancelled);
    uint32_t preview_calls = 0U;
    const interaction::SmartPickupPostStepResult post =
        controller.post_step(
            make_post_input(snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 0U && backend.observe_calls == 0U &&
            preview_calls == 0U && !post.pick_request.has_value(),
        "pending cancellation left activation work for post-step");
}

void test_pending_activation_yields_to_manual_override() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    require(
        activation.interact_consumed && backend.begin_calls == 0U,
        "pending-manual-override fixture did not capture its activation");

    interaction::SmartPickupPreStepInput override =
        make_pre_input(&target, false);
    override.manual_override_pressed = true;
    override.left_stick = vec3(-0.14F, 0.0F, 0.82F);
    override.right_stick = vec3(0.91F, 0.0F, -0.36F);
    override.force_strafe = true;
    const interaction::SmartPickupPreStepResult cancelled =
        controller.pre_step(override);
    require(
        cancelled.manual_override_consumed &&
            !cancelled.cancel_consumed &&
            !cancelled.interact_consumed &&
            same_vec3_bits(cancelled.left_stick, override.left_stick) &&
            same_vec3_bits(cancelled.right_stick, override.right_stick) &&
            cancelled.force_strafe == override.force_strafe &&
            backend.cancel_calls == 1U && backend.begin_calls == 0U,
        "pending Smart Pickup did not return manual control");

    CallerOrderHarness locomotion;
    const interaction::LocomotionSnapshot snapshot =
        locomotion.step(cancelled);
    uint32_t preview_calls = 0U;
    const interaction::SmartPickupPostStepResult post =
        controller.post_step(
            make_post_input(snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 0U && backend.observe_calls == 0U &&
            backend.take_submission_calls == 0U && preview_calls == 0U &&
            !post.pick_request.has_value(),
        "pending manual override left assist work for post-step");
}

void test_active_assist_yields_to_manual_override() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;
    locomotion.stick_displacement_scale = 0.0F;
    uint32_t preview_calls = 0U;

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    const interaction::LocomotionSnapshot activation_snapshot =
        locomotion.step(activation);
    const interaction::SmartPickupPostStepResult began =
        controller.post_step(
            make_post_input(activation_snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
            began.assist_output.override_steering && preview_calls == 0U,
        "active-manual-override fixture did not begin its assist");

    interaction::SmartPickupPreStepInput override =
        make_pre_input(&target, false);
    override.manual_override_pressed = true;
    override.left_stick = vec3(0.13F, 0.0F, -0.77F);
    override.right_stick = vec3(-0.42F, 0.0F, 0.19F);
    override.force_strafe = true;
    const interaction::SmartPickupPreStepResult cancelled =
        controller.pre_step(override);
    require(
        cancelled.manual_override_consumed &&
            !cancelled.cancel_consumed &&
            !cancelled.interact_consumed &&
            same_vec3_bits(cancelled.left_stick, override.left_stick) &&
            same_vec3_bits(cancelled.right_stick, override.right_stick) &&
            cancelled.force_strafe == override.force_strafe &&
            backend.cancel_calls == 1U && backend.begin_calls == 1U &&
            backend.observe_calls == 1U,
        "active Smart Pickup did not return manual control");

    const interaction::LocomotionSnapshot cancelled_snapshot =
        locomotion.step(cancelled);
    const interaction::SmartPickupPostStepResult cancelled_post =
        controller.post_step(
            make_post_input(cancelled_snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
            backend.take_submission_calls == 0U && preview_calls == 0U &&
            !cancelled_post.pick_request.has_value(),
        "active manual override produced post-step assist work");
}

void test_idle_interact_activation_wins_over_manual_override() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);

    interaction::SmartPickupPreStepInput input =
        make_pre_input(&target, true);
    input.manual_override_pressed = true;
    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(input);
    require(
        activation.interact_consumed &&
            !activation.cancel_consumed &&
            !activation.manual_override_consumed &&
            is_zero(activation.left_stick) &&
            is_zero(activation.right_stick) &&
            !activation.force_strafe && backend.cancel_calls == 0U &&
            backend.begin_calls == 0U,
        "idle simultaneous F and movement did not give activation precedence");

    CallerOrderHarness locomotion;
    const interaction::LocomotionSnapshot snapshot =
        locomotion.step(activation);
    uint32_t preview_calls = 0U;
    const interaction::SmartPickupPostStepResult post =
        controller.post_step(
            make_post_input(snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
            backend.take_submission_calls == 0U && preview_calls == 0U &&
            !post.pick_request.has_value(),
        "idle simultaneous F and movement cancelled the new activation");
}

void test_manual_override_passes_through_without_existing_ownership() {
    interaction::InteractionTarget target = make_controller_target();
    {
        CountingAssistBackend backend;
        interaction::SmartPickupController controller(backend);
        interaction::SmartPickupPreStepInput input =
            make_pre_input(&target, false);
        input.manual_override_pressed = true;
        input.force_strafe = true;
        const interaction::SmartPickupPreStepResult result =
            controller.pre_step(input);
        require(
            !result.manual_override_consumed &&
                !result.cancel_consumed && !result.interact_consumed &&
                same_vec3_bits(result.left_stick, input.left_stick) &&
                same_vec3_bits(result.right_stick, input.right_stick) &&
                result.force_strafe == input.force_strafe &&
                backend.cancel_calls == 0U &&
                controller.diagnostics().state ==
                    interaction::PickAssistState::Idle,
            "idle pickup assist consumed manual movement");
    }
    {
        InactiveStateAssistBackend backend(
            interaction::PickAssistState::Failed);
        interaction::SmartPickupController controller(backend);
        interaction::SmartPickupPreStepInput input =
            make_pre_input(&target, false);
        input.manual_override_pressed = true;
        input.force_strafe = true;
        const interaction::SmartPickupPreStepResult result =
            controller.pre_step(input);
        require(
            !result.manual_override_consumed &&
                !result.cancel_consumed && !result.interact_consumed &&
                same_vec3_bits(result.left_stick, input.left_stick) &&
                same_vec3_bits(result.right_stick, input.right_stick) &&
                result.force_strafe == input.force_strafe &&
                backend.cancel_calls == 0U &&
                controller.diagnostics().state ==
                    interaction::PickAssistState::Failed,
            "failed inactive pickup assist consumed manual movement");
    }
    {
        InactiveStateAssistBackend backend(
            interaction::PickAssistState::Submitted);
        interaction::SmartPickupController controller(backend);
        interaction::SmartPickupPreStepInput input =
            make_pre_input(&target, false);
        input.manual_override_pressed = true;
        input.force_strafe = true;
        const interaction::SmartPickupPreStepResult result =
            controller.pre_step(input);
        require(
            !result.manual_override_consumed &&
                !result.cancel_consumed && !result.interact_consumed &&
                same_vec3_bits(result.left_stick, input.left_stick) &&
                same_vec3_bits(result.right_stick, input.right_stick) &&
                result.force_strafe == input.force_strafe &&
                backend.cancel_calls == 0U &&
                controller.diagnostics().state ==
                    interaction::PickAssistState::Submitted,
            "submitted inactive pickup assist consumed manual movement");
    }
}

void test_simultaneous_cancel_wins_without_pending_or_observation() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    interaction::SmartPickupPreStepInput input =
        make_pre_input(&target, true, true);
    const interaction::SmartPickupPreStepResult pre =
        controller.pre_step(input);
    require(
        pre.cancel_consumed && pre.interact_consumed &&
            backend.cancel_calls == 1U && backend.begin_calls == 0U &&
            backend.observe_calls == 0U,
        "simultaneous X+F did not apply cancel precedence");
    require(
        same_vec3_bits(pre.left_stick, input.left_stick) &&
            same_vec3_bits(pre.right_stick, input.right_stick) &&
            pre.force_strafe == input.force_strafe,
        "idle cancel precedence unexpectedly owned manual movement");

    CallerOrderHarness locomotion;
    const interaction::LocomotionSnapshot snapshot = locomotion.step(pre);
    uint32_t preview_calls = 0U;
    const interaction::SmartPickupPostStepResult post =
        controller.post_step(
            make_post_input(snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 0U && backend.observe_calls == 0U &&
            backend.take_submission_calls == 0U && preview_calls == 0U &&
            !post.pick_request.has_value() &&
            controller.diagnostics().state ==
                interaction::PickAssistState::Idle &&
            controller.diagnostics().reason ==
                interaction::PickAssistReason::Cancelled,
        "cancelled simultaneous activation created assist work");
}

void test_active_assist_owns_sticks_while_camera_and_cancel_remain_live() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;
    locomotion.stick_displacement_scale = 0.0F;
    uint32_t preview_calls = 0U;

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    const interaction::LocomotionSnapshot first_snapshot =
        locomotion.step(activation);
    interaction::SmartPickupPostStepInput first_post =
        make_post_input(first_snapshot, &target);
    first_post.camera_azimuth = 0.0F;
    const interaction::SmartPickupPostStepResult first =
        controller.post_step(
            first_post, rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
            first.assist_output.override_steering &&
            !is_zero(first.assist_output.left_stick),
        "camera/cancel fixture did not enter active slot approach");

    interaction::SmartPickupPreStepInput raw =
        make_pre_input(&target, false);
    raw.left_stick = vec3(-0.91F, 0.0F, 0.37F);
    raw.right_stick = vec3(0.28F, 0.0F, -0.83F);
    const interaction::SmartPickupPreStepResult owned =
        controller.pre_step(raw);
    require(
        same_vec3_bits(owned.left_stick, first.assist_output.left_stick) &&
            same_vec3_bits(
                owned.right_stick, first.assist_output.right_stick) &&
            !same_vec3_bits(owned.left_stick, raw.left_stick),
        "active manual assist did not replace raw character sticks");

    const interaction::LocomotionSnapshot second_snapshot =
        locomotion.step(owned);
    interaction::SmartPickupPostStepInput second_post =
        make_post_input(second_snapshot, &target);
    second_post.camera_azimuth = 1.10F;
    const interaction::SmartPickupPostStepResult second =
        controller.post_step(
            second_post, rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 1U && backend.observe_calls == 2U &&
            second.assist_output.override_steering &&
            !same_vec3_bits(
                second.assist_output.left_stick,
                first.assist_output.left_stick),
        "new camera azimuth did not update camera-relative assist steering");

    interaction::SmartPickupPreStepInput cancel_input =
        make_pre_input(&target, false, true);
    cancel_input.left_stick = vec3(0.13F, 0.0F, -0.77F);
    cancel_input.right_stick = vec3(-0.42F, 0.0F, 0.19F);
    const interaction::SmartPickupPreStepResult cancelled =
        controller.pre_step(cancel_input);
    require(
        cancelled.cancel_consumed && !cancelled.interact_consumed &&
            backend.cancel_calls == 1U &&
            same_vec3_bits(
                cancelled.left_stick, cancel_input.left_stick) &&
            same_vec3_bits(
                cancelled.right_stick, cancel_input.right_stick) &&
            controller.diagnostics().state ==
                interaction::PickAssistState::Idle &&
            controller.diagnostics().reason ==
                interaction::PickAssistReason::Cancelled,
        "active cancel was not responsive in the same pre-step");

    const interaction::LocomotionSnapshot cancelled_snapshot =
        locomotion.step(cancelled);
    const interaction::SmartPickupPostStepResult cancelled_post =
        controller.post_step(
            make_post_input(cancelled_snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 1U && backend.observe_calls == 2U &&
            backend.take_submission_calls == 0U && preview_calls == 0U &&
            !cancelled_post.pick_request.has_value(),
        "cancelled active assist produced post-step work");
}

void test_active_repeated_f_is_consumed_without_restarting_assist() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;
    locomotion.stick_displacement_scale = 0.0F;
    uint32_t preview_calls = 0U;

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    const interaction::LocomotionSnapshot activation_snapshot =
        locomotion.step(activation);
    const interaction::SmartPickupPostStepResult first =
        controller.post_step(
            make_post_input(activation_snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
            first.assist_output.override_steering,
        "repeated-F fixture did not activate once");

    interaction::SmartPickupPreStepInput repeated_input =
        make_pre_input(&target, true);
    repeated_input.left_stick = vec3(-0.85F, 0.0F, 0.24F);
    repeated_input.right_stick = vec3(0.61F, 0.0F, -0.48F);
    const interaction::SmartPickupPreStepResult repeated =
        controller.pre_step(repeated_input);
    require(
        repeated.interact_consumed && !repeated.cancel_consumed &&
            same_vec3_bits(
                repeated.left_stick, first.assist_output.left_stick) &&
            same_vec3_bits(
                repeated.right_stick, first.assist_output.right_stick) &&
            backend.begin_calls == 1U && backend.observe_calls == 1U,
        "active repeated F escaped consumption or restarted in pre_step");

    const interaction::LocomotionSnapshot repeated_snapshot =
        locomotion.step(repeated);
    const interaction::SmartPickupPostStepResult second =
        controller.post_step(
            make_post_input(repeated_snapshot, &target),
            rejecting_preview_counter(preview_calls));
    require(
        locomotion.ordinary_step_calls == 2U &&
            locomotion.materialized_snapshot_count == 2U &&
            backend.begin_calls == 1U && backend.observe_calls == 2U &&
            backend.take_submission_calls == 0U && preview_calls == 0U &&
            !second.pick_request.has_value() &&
            controller.diagnostics().state ==
                interaction::PickAssistState::SlotApproach,
        "active repeated F created a second begin, preview, or request");
}

void test_post_obstacles_are_copied_into_backend_and_frozen() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;
    locomotion.stick_displacement_scale = 0.0F;
    const std::vector<vec3> obstacle_centers{
        vec3(9.25F, 3.50F, -7.75F),
        vec3(0.0F, 0.0F, 0.0F),
    };
    const std::vector<vec3> obstacle_sizes{
        vec3(0.25F, 0.50F, 0.75F),
        vec3(0.01F, 0.01F, 0.01F),
    };
    const std::vector<vec3> centers_before = obstacle_centers;
    const std::vector<vec3> sizes_before = obstacle_sizes;

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    const interaction::LocomotionSnapshot snapshot =
        locomotion.step(activation);
    uint32_t preview_calls = 0U;
    const interaction::SmartPickupPostStepResult began =
        controller.post_step(
            make_post_input(
                snapshot, &target, obstacle_centers, obstacle_sizes),
            rejecting_preview_counter(preview_calls));
    const interaction::PickAssistDiagnostics initial =
        controller.diagnostics();
    require(
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
            !began.pick_request.has_value() &&
            backend.begin_inputs.size() == 1U &&
            same_navigation_obstacle_bits(
                backend.begin_inputs.front().obstacles,
                obstacle_centers,
                obstacle_sizes) &&
            same_vec3_vector_bits(obstacle_centers, centers_before) &&
            same_vec3_vector_bits(obstacle_sizes, sizes_before),
        "coordinator did not copy ordered post-step obstacles exactly");
    require(
        initial.selected_slot_id == 22U &&
            initial.slot_selection.ordered.size() == 3U &&
            initial.slot_selection.ordered[0].id == 11U &&
            initial.slot_selection.ordered[0].reason ==
                interaction::PickSlotReason::ObstacleBlocked &&
            initial.slot_selection.ordered[0].obstacle_index == 1,
        "ordered obstacle center/size conversion did not block only slot 11");

    const std::vector<vec3> replacement_centers{
        vec3(-0.40F, 0.0F, -0.50F),
    };
    const std::vector<vec3> replacement_sizes{
        vec3(0.10F, 0.10F, 0.10F),
    };
    const interaction::SmartPickupPreStepResult active =
        controller.pre_step(make_pre_input(&target, false));
    const interaction::LocomotionSnapshot second_snapshot =
        locomotion.step(active);
    const interaction::SmartPickupPostStepResult observed =
        controller.post_step(
            make_post_input(
                second_snapshot,
                &target,
                replacement_centers,
                replacement_sizes),
            rejecting_preview_counter(preview_calls));
    const interaction::PickAssistDiagnostics& frozen =
        controller.diagnostics();
    require(
        backend.begin_calls == 1U && backend.observe_calls == 2U &&
            backend.take_submission_calls == 0U && preview_calls == 0U &&
            !observed.pick_request.has_value() &&
            frozen.state == interaction::PickAssistState::SlotApproach &&
            frozen.reason == interaction::PickAssistReason::None &&
            frozen.selected_slot_id == initial.selected_slot_id &&
            frozen.slot_selection.selected_index ==
                initial.slot_selection.selected_index &&
            frozen.slot_selection.ordered[0].obstacle_index == 1,
        "later controller obstacles replaced the attempt's frozen copy");
}

interaction::PickEntryPreview make_certified_preview(
    interaction::PickEntryRoot root) {
    interaction::PickEntryPreview preview{};
    preview.path_feasible = true;
    preview.match_ready = true;
    preview.path_reason = interaction::Reason::None;
    preview.match_reason = interaction::Reason::None;
    preview.prospective_root = root;
    preview.feasible_entry_frame = 125;
    preview.contact_frame = 150;
    preview.total_cost = 4.25F;
    return preview;
}

bool is_stationary_preview_retry(
    const interaction::PickAssistOutput& output,
    interaction::PickEntryRoot root) {
    return output.override_steering && output.force_strafe &&
        output.stationary_constraint && is_zero(output.left_stick) &&
        is_zero(output.right_stick) && output.needs_preview &&
        output.preview_root.has_value() &&
        same_pick_entry_root_bits(*output.preview_root, root) &&
        !output.submit_interact;
}

void test_null_and_poor_match_final_preview_retry_stationary() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;
    locomotion.root_position = vec3(0.0F, 0.0F, 0.0F);
    locomotion.root_yaw_radians = 0.0F;
    locomotion.stick_displacement_scale = 0.0F;
    const interaction::PickEntryRoot expected_root{0.0F, 0.0F, 0.0F};
    enum class PreviewMode { Missing, PoorMatch };
    PreviewMode mode = PreviewMode::Missing;
    uint32_t preview_calls = 0U;
    const interaction::SmartPickupPreviewCallback preview =
        [&](const interaction::LocomotionSnapshot&,
            interaction::PickEntryRoot root,
            interaction::TargetHandle handle,
            uint32_t affordance_id)
            -> std::optional<interaction::PickEntryPreview> {
            ++preview_calls;
            require(
                same_pick_entry_root_bits(root, expected_root) &&
                    handle == target.handle &&
                    affordance_id == target.affordances.front().id,
                "retry preview did not receive frozen target/slot identity");
            if (mode == PreviewMode::Missing) return std::nullopt;
            interaction::PickEntryPreview poor{};
            poor.path_feasible = true;
            poor.path_reason = interaction::Reason::None;
            poor.match_ready = false;
            poor.match_reason = interaction::Reason::PoorMatch;
            poor.prospective_root = root;
            return poor;
        };

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    const interaction::LocomotionSnapshot activation_snapshot =
        locomotion.step(activation);
    interaction::SmartPickupPostStepResult post = controller.post_step(
        make_post_input(activation_snapshot, &target), preview);
    require(
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
            controller.diagnostics().state ==
                interaction::PickAssistState::Settling &&
            preview_calls == 0U && !post.pick_request.has_value(),
        "retry fixture did not activate at the exact frozen slot");

    for (uint32_t tick = 1U; tick <= 5U; ++tick) {
        const interaction::SmartPickupPreStepResult pre =
            controller.pre_step(make_pre_input(&target, false));
        const interaction::LocomotionSnapshot snapshot =
            locomotion.step(pre);
        post = controller.post_step(
            make_post_input(snapshot, &target), preview);
    }
    require(
        backend.begin_calls == 1U && backend.observe_calls == 6U &&
            preview_calls == 0U &&
            controller.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            is_stationary_preview_retry(post.assist_output, expected_root),
        "retry fixture did not enter stationary FinalPreview after five ticks");
    const uint32_t frozen_slot_id =
        controller.diagnostics().selected_slot_id;
    const std::optional<size_t> frozen_slot_index =
        controller.diagnostics().slot_selection.selected_index;

    const interaction::SmartPickupPreStepResult missing_pre =
        controller.pre_step(make_pre_input(&target, false));
    const interaction::LocomotionSnapshot missing_snapshot =
        locomotion.step(missing_pre);
    const interaction::SmartPickupPostStepResult missing =
        controller.post_step(
            make_post_input(missing_snapshot, &target), preview);
    require(
        preview_calls == 1U && backend.begin_calls == 1U &&
            backend.observe_calls == 7U &&
            backend.take_submission_calls == 0U &&
            !missing.pick_request.has_value() &&
            controller.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            controller.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            is_stationary_preview_retry(
                missing.assist_output, expected_root),
        "null preview did not remain stationary and request one retry");

    mode = PreviewMode::PoorMatch;
    const interaction::SmartPickupPreStepResult poor_pre =
        controller.pre_step(make_pre_input(&target, false));
    const interaction::LocomotionSnapshot poor_snapshot =
        locomotion.step(poor_pre);
    const interaction::SmartPickupPostStepResult poor =
        controller.post_step(
            make_post_input(poor_snapshot, &target), preview);
    require(
        preview_calls == 2U && backend.begin_calls == 1U &&
            backend.observe_calls == 8U &&
            backend.take_submission_calls == 0U &&
            !poor.pick_request.has_value() &&
            controller.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            controller.diagnostics().reason ==
                interaction::PickAssistReason::None &&
            controller.diagnostics().final_preview.match_reason ==
                interaction::Reason::PoorMatch &&
            controller.diagnostics().selected_slot_id == frozen_slot_id &&
            controller.diagnostics().slot_selection.selected_index ==
                frozen_slot_index &&
            is_stationary_preview_retry(poor.assist_output, expected_root),
        "PoorMatch preview did not retry stationarily on the frozen slot");
}

void test_final_preview_and_request_are_each_one_shot() {
    interaction::InteractionTarget target = make_controller_target();
    CountingAssistBackend backend;
    interaction::SmartPickupController controller(backend);
    CallerOrderHarness locomotion;
    locomotion.root_position = vec3(0.0F, 0.0F, 0.0F);
    locomotion.root_yaw_radians = 0.0F;
    locomotion.stick_displacement_scale = 0.0F;
    uint32_t preview_calls = 0U;
    uint64_t callback_snapshot_fingerprint = 0U;
    interaction::LocomotionSnapshot expected_preview_snapshot{};
    bool expected_preview_snapshot_available = false;
    const interaction::PickEntryRoot expected_root{0.0F, 0.0F, 0.0F};
    const interaction::SmartPickupPreviewCallback preview =
        [&](const interaction::LocomotionSnapshot& snapshot,
            interaction::PickEntryRoot root,
            interaction::TargetHandle handle,
            uint32_t affordance_id)
            -> std::optional<interaction::PickEntryPreview> {
            ++preview_calls;
            require(
                expected_preview_snapshot_available &&
                    same_snapshot_bits(
                        snapshot, expected_preview_snapshot),
                "preview did not receive the current authoritative snapshot");
            require(
                same_pick_entry_root_bits(root, expected_root) &&
                    handle == target.handle &&
                    affordance_id == target.affordances.front().id,
                "preview did not receive the frozen slot identity");
            callback_snapshot_fingerprint =
                interaction::runtime_detail::
                    locomotion_snapshot_fingerprint(snapshot);
            return make_certified_preview(root);
        };

    const interaction::SmartPickupPreStepResult activation =
        controller.pre_step(make_pre_input(&target, true));
    const interaction::LocomotionSnapshot activation_snapshot =
        locomotion.step(activation);
    interaction::SmartPickupPostStepResult post = controller.post_step(
        make_post_input(activation_snapshot, &target), preview);
    require(
        backend.begin_calls == 1U && backend.observe_calls == 1U &&
            backend.take_submission_calls == 0U && preview_calls == 0U &&
            !post.pick_request.has_value() &&
            controller.diagnostics().state ==
                interaction::PickAssistState::Settling,
        "exact-slot activation did not enter Settling without previewing");

    for (uint32_t tick = 1U; tick <= 5U; ++tick) {
        const interaction::SmartPickupPreStepResult pre =
            controller.pre_step(make_pre_input(&target, false));
        const interaction::LocomotionSnapshot snapshot = locomotion.step(pre);
        post = controller.post_step(make_post_input(snapshot, &target), preview);
        require(
            backend.begin_calls == 1U &&
                backend.observe_calls == tick + 1U &&
                backend.take_submission_calls == 0U &&
                preview_calls == 0U &&
                !post.pick_request.has_value() &&
                controller.diagnostics().settle_ticks == tick,
            "settling performed hidden begin, preview, request, or observation work");
        if (tick < 5U) {
            require(
                controller.diagnostics().state ==
                    interaction::PickAssistState::Settling,
                "settling completed before five 25 Hz ticks");
        }
    }
    require(
        controller.diagnostics().state ==
                interaction::PickAssistState::FinalPreview &&
            post.assist_output.needs_preview &&
            post.assist_output.preview_root.has_value() &&
            same_pick_entry_root_bits(
                *post.assist_output.preview_root, expected_root),
        "fifth settle tick did not request the exact frozen preview root");

    const interaction::SmartPickupPreStepResult preview_pre =
        controller.pre_step(make_pre_input(&target, false));
    expected_preview_snapshot = locomotion.step(preview_pre);
    expected_preview_snapshot_available = true;
    post = controller.post_step(
        make_post_input(expected_preview_snapshot, &target), preview);
    require(
        preview_calls == 1U && backend.begin_calls == 1U &&
            backend.observe_calls == 7U &&
            backend.take_submission_calls == 1U &&
            backend.yielded_submissions == 1U &&
            post.assist_output.submit_interact &&
            post.pick_request.has_value() &&
            post.pick_request->target == target.handle &&
            post.pick_request->affordance_id ==
                target.affordances.front().id &&
            post.pick_request->request_id == 71U &&
            controller.diagnostics().state ==
                interaction::PickAssistState::Submitted &&
            post.snapshot_fingerprint == callback_snapshot_fingerprint,
        "certified preview did not yield exactly one matching PickRequest");

    const interaction::SmartPickupPreStepResult after_submission =
        controller.pre_step(make_pre_input(&target, false));
    const interaction::LocomotionSnapshot after_snapshot =
        locomotion.step(after_submission);
    const interaction::SmartPickupPostStepResult after =
        controller.post_step(
            make_post_input(after_snapshot, &target), preview);
    require(
        preview_calls == 1U && backend.begin_calls == 1U &&
            backend.observe_calls == 7U &&
            backend.take_submission_calls == 1U &&
            backend.yielded_submissions == 1U &&
            !after.pick_request.has_value() &&
            !after.assist_output.submit_interact,
        "Submitted state repeated preview or request extraction");
}

}  // namespace

int main() {
    test_activation_brackets_one_caller_step_and_defers_assist_motion();
    test_begin_exception_clears_pending_activation_before_backend_call();
    test_obstacle_array_exception_clears_pending_activation();
    test_pending_target_mutations_fail_before_slot_freeze();
    test_post_begin_slot_mutations_fail_without_hopping();
    test_missing_target_and_selector_fail_stably_after_consuming_f();
    test_target_disappearing_during_step_is_observed_once_and_fails();
    test_cancel_passes_through_without_smart_pickup_ownership();
    test_pending_activation_owns_cancel_edge();
    test_pending_activation_yields_to_manual_override();
    test_active_assist_yields_to_manual_override();
    test_idle_interact_activation_wins_over_manual_override();
    test_manual_override_passes_through_without_existing_ownership();
    test_simultaneous_cancel_wins_without_pending_or_observation();
    test_active_assist_owns_sticks_while_camera_and_cancel_remain_live();
    test_active_repeated_f_is_consumed_without_restarting_assist();
    test_post_obstacles_are_copied_into_backend_and_frozen();
    test_null_and_poor_match_final_preview_retry_stationary();
    test_final_preview_and_request_are_each_one_shot();
    return 0;
}
