#include "interaction_learned_pickup_backend.h"

#include "interaction_funnel_artifact.h"
#include "interaction_learned_pickup_diagnostics.h"
#include "interaction_smart_pickup_controller.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {

using interaction::FunnelCancelReason;
using interaction::FunnelFollowerState;
using interaction::InteractionFunnelArtifact;
using interaction::LearnedSmartPickupBackend;
using interaction::PickAssistReason;
using interaction::PickAssistState;
using interaction::SmartPickupAssistBackend;
using interaction::kFunnelConditionDim;
using interaction::kFunnelExecutionTickCount;
using interaction::kFunnelKnotCount;
using interaction::kFunnelProposalCount;
using interaction::kFunnelSampleCount;
using interaction::kFunnelSampleWidth;

// The learned backend must be usable wherever the authored backend is.
static_assert(std::is_base_of_v<SmartPickupAssistBackend, LearnedSmartPickupBackend>);
static_assert(std::has_virtual_destructor_v<SmartPickupAssistBackend>);

constexpr char kMagic[8] = {'G', '1', 'F', 'U', 'N', 'N', 'L', '3'};

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFFU));
}
void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) bytes.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFFU));
}
void append_f32(std::vector<uint8_t>& bytes, float value) {
    uint32_t raw = 0U;
    std::memcpy(&raw, &value, sizeof(raw));
    append_u32(bytes, raw);
}

void append_identity(
    std::vector<uint8_t>& bytes,
    const interaction::FunnelProposalRequest* request = nullptr) {
    append_u64(bytes, request == nullptr ? 7U : request->request_id);
    append_u64(bytes, request == nullptr ? 2026071901U : request->batch_seed);
    if (request == nullptr) {
        for (uint8_t value = 0U; value < 32U; ++value) bytes.push_back(value);
    } else {
        for (uint8_t value : request->checkpoint_sha256) bytes.push_back(value);
    }
}

struct SampleTuple {
    float x, z, sin, cos;
};

// Build a valid artifact where proposal 0 is rejected-and-discontinuous. Static
// seam tests use +x motion; provider tests converge from the conditioned entry
// to one fixed object-local interaction root.
InteractionFunnelArtifact make_artifact(
    const interaction::FunnelProposalRequest* request = nullptr,
    bool any_accepted = true) {
    std::vector<uint8_t> bytes;
    for (char c : kMagic) bytes.push_back(static_cast<uint8_t>(c));
    append_u32(bytes, 3U);
    append_u32(bytes, static_cast<uint32_t>(kFunnelConditionDim));
    append_u32(bytes, static_cast<uint32_t>(kFunnelProposalCount));
    append_u32(bytes, static_cast<uint32_t>(kFunnelSampleCount));
    append_u32(bytes, static_cast<uint32_t>(kFunnelSampleWidth));
    append_identity(bytes, request);
    std::array<float, kFunnelConditionDim> condition{};
    condition[21] = 1.0F;
    if (request != nullptr) condition = request->condition;
    for (float value : condition) append_f32(bytes, value);
    for (int p = 0; p < kFunnelProposalCount; ++p) {
        for (int s = 0; s < kFunnelSampleCount; ++s) {
            const float alpha = static_cast<float>(s) /
                static_cast<float>(kFunnelSampleCount - 1);
            SampleTuple sample{};
            if (request == nullptr) {
                sample = {
                    0.04F * static_cast<float>(s), 0.0F, 0.0F, 1.0F};
            } else {
                const float terminal_z = condition[19] < 0.0F
                    ? -0.94F
                    : 0.94F;
                sample = {
                    condition[18] + alpha * (0.18F - condition[18]),
                    condition[19] + alpha * (terminal_z - condition[19]),
                    condition[20],
                    condition[21],
                };
                if (s == 0) {
                    sample = {condition[18], condition[19],
                              condition[20], condition[21]};
                }
            }
            if (p == 0 && s == 8) sample.x = 5.0F;  // discontinuous, only ok if rejected
            append_f32(bytes, sample.x);
            append_f32(bytes, sample.z);
            append_f32(bytes, sample.sin);
            append_f32(bytes, sample.cos);
        }
    }
    for (int p = 0; p < kFunnelProposalCount; ++p) append_u64(bytes, static_cast<uint64_t>(100 + p));
    for (int p = 0; p < kFunnelProposalCount; ++p) {
        bytes.push_back(any_accepted && p != 0 ? 1U : 0U);
    }
    return InteractionFunnelArtifact::load(bytes);
}

interaction::FunnelSample sample_at(float x) {
    interaction::FunnelSample s{};
    s.x = x;
    s.z = 0.0F;
    s.yaw_sin = 0.0F;
    s.yaw_cos = 1.0F;
    return s;
}

class FakeProposalProvider final : public interaction::FunnelProposalProvider {
public:
    bool begin(const interaction::FunnelProposalRequest& value) override {
        ++begin_calls;
        request = value;
        active = true;
        return begin_succeeds;
    }

    interaction::FunnelProposalPoll poll() override {
        ++poll_calls;
        if (!active || !ready) return {};
        active = false;
        return {
            interaction::FunnelProposalPollState::Ready,
            make_artifact(&request, any_accepted),
            {},
        };
    }

    void cancel() override {
        ++cancel_calls;
        active = false;
    }

    bool begin_succeeds = true;
    bool ready = false;
    bool active = false;
    bool any_accepted = true;
    int begin_calls = 0;
    int poll_calls = 0;
    int cancel_calls = 0;
    interaction::FunnelProposalRequest request{};
};

interaction::InteractionTarget lifecycle_target() {
    interaction::InteractionTarget target{};
    target.handle = {71U, 9U};
    target.object_world = {
        vec3(0.0F, 0.8F, 0.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    target.object_dimensions = vec3(0.1F, 0.2F, 0.1F);
    target.object_bounds.half_extents_object = vec3(0.05F, 0.1F, 0.05F);
    target.table_world = {
        vec3(100.0F, 0.0F, 100.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    target.table_size = vec3(1.0F, 0.1F, 1.0F);
    interaction::GraspAffordance affordance{};
    affordance.id = 7U;
    affordance.hand = interaction::Hand::Right;
    affordance.hand_in_object = {
        vec3(0.0F, 0.1F, 0.0F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    affordance.approach_direction_object = vec3(0.0F, 0.0F, 1.0F);
    target.affordances.push_back(affordance);
    return target;
}

interaction::Transform lifecycle_entry() {
    return {
        vec3(0.0F, 0.0F, 1.0F),
        quat_from_angle_axis(PIf, vec3(0.0F, 1.0F, 0.0F)),
    };
}

interaction::PickAssistObservation lifecycle_observation(
    uint64_t tick,
    const interaction::InteractionTarget& target,
    interaction::Transform root) {
    interaction::PickAssistObservation observation{};
    observation.controller_tick = tick;
    observation.runtime_state = interaction::RuntimeState::Locomotion;
    observation.target = &target;
    observation.displayed_root = root;
    observation.snapshot_fingerprint = tick + 1000U;
    return observation;
}

void begin_and_settle(
    LearnedSmartPickupBackend& backend,
    const interaction::InteractionTarget& target,
    interaction::Transform entry) {
    interaction::PickAssistStart start{};
    start.target_snapshot = target;
    start.affordance_id = 7U;
    start.root_world = entry;
    start.controller_tick = 10U;
    assert(backend.begin(start, &target));
    backend.observe(lifecycle_observation(10U, target, entry));
    backend.observe(lifecycle_observation(11U, target, entry));
    backend.observe(lifecycle_observation(12U, target, entry));
}

void test_reports_accepted_proposal_count() {
    LearnedSmartPickupBackend backend(make_artifact());
    assert(backend.learned_diagnostics().accepted_proposal_count == kFunnelProposalCount - 1);
    assert(!backend.learned_diagnostics().armed);
}

void test_delegates_begin_failure_to_authored_controller() {
    // Preserve authored behavior: a begin with no post-step target fails exactly
    // as the authored ControllerPickAssist does (TargetUnavailable).
    LearnedSmartPickupBackend backend(make_artifact());
    interaction::PickAssistStart start{};
    bool began = backend.begin(start, nullptr);
    assert(!began);
    assert(!backend.active());
    assert(backend.diagnostics().state == PickAssistState::Failed);
    assert(backend.diagnostics().reason == PickAssistReason::TargetUnavailable);
}

void test_arm_selects_first_accepted_seed() {
    LearnedSmartPickupBackend backend(make_artifact());
    bool armed = backend.arm(200U);
    assert(armed);
    assert(backend.learned_diagnostics().armed);
    // Proposal 0 is rejected; proposal 1 (seed 101) is the first accepted.
    assert(backend.learned_diagnostics().armed_seed == 101U);
}

void test_follower_mirrors_spatial_lookahead_and_completion() {
    LearnedSmartPickupBackend backend(make_artifact());
    assert(backend.arm(0U));
    std::array<interaction::FunnelSample, kFunnelKnotCount> knots{};
    for (int knot = 0; knot < kFunnelKnotCount; ++knot) {
        knots[static_cast<size_t>(knot)] = sample_at(
            0.04F * static_cast<float>(knot));
    }
    const auto targets = interaction::expand_funnel_execution(knots);
    interaction::FunnelFollowerOutput first = backend.follow({0U, sample_at(0.0F)});
    assert(first.published);
    assert(first.sample.x >= interaction::kFunnelLookaheadDistance - 0.01F);
    assert(backend.follow({1U, targets.back()}).published);
    assert(backend.follow({2U, targets.back()}).published);
    assert(backend.learned_diagnostics().follower_state == FunnelFollowerState::Following);
    const auto completed = backend.follow({3U, targets.back()});
    assert(!completed.published);
    assert(backend.learned_diagnostics().follower_state == FunnelFollowerState::Completed);
    assert(backend.learned_diagnostics().follower_progress_index ==
           kFunnelExecutionTickCount - 1);
    assert(backend.learned_diagnostics().follower_lookahead_index ==
           kFunnelExecutionTickCount - 1);
}

void test_follow_before_arm_publishes_nothing() {
    LearnedSmartPickupBackend backend(make_artifact());
    interaction::FunnelFollowerOutput out = backend.follow({0U, sample_at(0.0F)});
    assert(!out.published);
    assert(!backend.learned_diagnostics().armed);
}

void test_missed_tick_cancellation_is_mirrored() {
    LearnedSmartPickupBackend backend(make_artifact());
    assert(backend.arm(10U));
    backend.follow({10U, sample_at(0.0F)});
    interaction::FunnelFollowerOutput out = backend.follow({12U, sample_at(0.0F)});
    assert(!out.published);
    assert(backend.learned_diagnostics().follower_state == FunnelFollowerState::Cancelled);
    assert(backend.learned_diagnostics().follower_cancel_reason == FunnelCancelReason::MissedTick);
}

void test_backend_cancel_terminates_the_learned_follower() {
    LearnedSmartPickupBackend backend(make_artifact());
    assert(backend.arm(10U));
    assert(backend.follow({10U, sample_at(0.0F)}).published);

    backend.cancel();

    assert(!backend.learned_diagnostics().armed);
    assert(backend.learned_diagnostics().follower_state == FunnelFollowerState::Cancelled);
    assert(backend.learned_diagnostics().follower_cancel_reason ==
           FunnelCancelReason::ExplicitCancel);
    interaction::FunnelFollowerOutput out = backend.follow({11U, sample_at(0.0F)});
    assert(!out.published);
    assert(out.state == FunnelFollowerState::Cancelled);
}

void test_arm_is_rejected_when_no_proposal_accepted() {
    // Build an artifact with every proposal rejected.
    std::vector<uint8_t> bytes;
    for (char c : kMagic) bytes.push_back(static_cast<uint8_t>(c));
    append_u32(bytes, 3U);
    append_u32(bytes, static_cast<uint32_t>(kFunnelConditionDim));
    append_u32(bytes, static_cast<uint32_t>(kFunnelProposalCount));
    append_u32(bytes, static_cast<uint32_t>(kFunnelSampleCount));
    append_u32(bytes, static_cast<uint32_t>(kFunnelSampleWidth));
    append_identity(bytes);
    for (int i = 0; i < kFunnelConditionDim; ++i) {
        append_f32(bytes, i == 21 ? 1.0F : 0.0F);
    }
    for (int p = 0; p < kFunnelProposalCount; ++p) {
        for (int s = 0; s < kFunnelSampleCount; ++s) {
            append_f32(bytes, 0.04F * static_cast<float>(s));
            append_f32(bytes, 0.0F);
            append_f32(bytes, 0.0F);
            append_f32(bytes, 1.0F);
        }
    }
    for (int p = 0; p < kFunnelProposalCount; ++p) append_u64(bytes, static_cast<uint64_t>(p));
    for (int p = 0; p < kFunnelProposalCount; ++p) bytes.push_back(0U);
    LearnedSmartPickupBackend backend(InteractionFunnelArtifact::load(bytes));
    assert(backend.learned_diagnostics().accepted_proposal_count == 0);
    assert(!backend.arm(0U));
    assert(!backend.learned_diagnostics().armed);
}

void test_provider_backed_lifecycle_submits_after_native_funnel() {
    FakeProposalProvider provider;
    interaction::LearnedPickupConfig config{};
    for (size_t index = 0U; index < config.checkpoint_sha256.size(); ++index) {
        config.checkpoint_sha256[index] = static_cast<uint8_t>(index);
    }
    LearnedSmartPickupBackend backend(provider, config);
    const interaction::InteractionTarget target = lifecycle_target();
    const interaction::Transform entry = lifecycle_entry();
    interaction::PickAssistStart start{};
    start.target_snapshot = target;
    start.affordance_id = 7U;
    start.root_world = entry;
    start.controller_tick = 10U;
    assert(backend.begin(start, &target));
    assert(backend.learned_diagnostics().state ==
           interaction::LearnedPickupState::CoarseCapture);

    assert(backend.observe(lifecycle_observation(10U, target, entry))
               .stationary_constraint);
    assert(backend.observe(lifecycle_observation(11U, target, entry))
               .stationary_constraint);
    assert(backend.observe(lifecycle_observation(12U, target, entry))
               .stationary_constraint);
    assert(provider.begin_calls == 1);
    assert(backend.learned_diagnostics().state ==
           interaction::LearnedPickupState::ProposalPending);

    assert(backend.observe(lifecycle_observation(13U, target, entry))
               .stationary_constraint);
    provider.ready = true;
    const interaction::PickAssistOutput selection =
        backend.observe(lifecycle_observation(14U, target, entry));
    assert(selection.preview_requests.size() == 31U);
    assert(backend.learned_diagnostics().state ==
           interaction::LearnedPickupState::SelectionPreview);

    interaction::PickAssistObservation selected =
        lifecycle_observation(15U, target, entry);
    selected.preview_snapshot_fingerprint = selected.snapshot_fingerprint;
    for (const interaction::PickAssistPreviewRequest& request :
         selection.preview_requests) {
        interaction::PickEntryPreview preview{};
        preview.path_feasible = true;
        preview.match_ready = true;
        preview.prospective_root = request.root;
        preview.total_cost = static_cast<float>(request.slot_id);
        selected.preview_results.push_back({request, preview});
    }
    assert(backend.observe(selected).stationary_constraint);
    assert(backend.learned_diagnostics().state ==
           interaction::LearnedPickupState::FunnelFollow);
    assert(backend.learned_diagnostics().selected_proposal_index == 1);
    const auto debug = backend.learned_debug_snapshot();
    assert(debug.has_value());
    assert(debug->active);
    assert(std::abs(debug->world_route.front().x - entry.position.x) < 1.0e-5F);
    assert(std::abs(debug->world_route.front().z - entry.position.z) < 1.0e-5F);
    assert(std::abs(debug->terminal_root_world.x - 0.18F) < 1.0e-5F);
    assert(std::abs(debug->terminal_root_world.z - 0.94F) < 1.0e-5F);

    interaction::Transform tracked{
        vec3(0.18F, entry.position.y, 0.94F),
        entry.rotation,
    };
    const interaction::PickAssistOutput route_follow =
        backend.observe(lifecycle_observation(16U, target, tracked));
    assert(route_follow.override_steering);
    assert(route_follow.force_strafe);
    assert(backend.observe(lifecycle_observation(17U, target, tracked))
               .override_steering);
    const interaction::PickAssistOutput final_request = backend.observe(
        lifecycle_observation(18U, target, tracked));
    assert(final_request.preview_requests.size() == 1U);
    assert(backend.learned_diagnostics().state ==
           interaction::LearnedPickupState::FinalPreview);
    assert(backend.learned_diagnostics().follower_progress_index ==
           interaction::kFunnelExecutionTickCount - 1);
    assert(backend.learned_diagnostics().follower_lookahead_index ==
           interaction::kFunnelExecutionTickCount - 1);

    interaction::PickAssistObservation final =
        lifecycle_observation(19U, target, tracked);
    final.preview_snapshot_fingerprint = final.snapshot_fingerprint;
    interaction::PickEntryPreview preview{};
    preview.path_feasible = true;
    preview.match_ready = true;
    preview.prospective_root = final_request.preview_requests.front().root;
    final.preview_results.push_back({
        final_request.preview_requests.front(), preview});
    const interaction::PickAssistOutput ready = backend.observe(final);
    assert(ready.submit_interact);
    const std::optional<interaction::PickRequest> request =
        backend.take_submission(9001U);
    assert(request.has_value());
    assert(request->target == target.handle);
    assert(request->affordance_id == 7U);
    assert(request->request_id == 9001U);
    assert(backend.learned_diagnostics().state ==
           interaction::LearnedPickupState::Submitted);
}

void test_provider_timeout_fails_closed_and_cancels_worker() {
    FakeProposalProvider provider;
    interaction::LearnedPickupConfig config{};
    config.maximum_proposal_pending_ticks = 1U;
    LearnedSmartPickupBackend backend(provider, config);
    const auto target = lifecycle_target();
    const auto entry = lifecycle_entry();
    begin_and_settle(backend, target, entry);
    backend.observe(lifecycle_observation(13U, target, entry));
    backend.observe(lifecycle_observation(14U, target, entry));
    assert(backend.learned_diagnostics().state ==
           interaction::LearnedPickupState::Failed);
    assert(backend.learned_diagnostics().failure_reason ==
           interaction::LearnedPickupFailureReason::ProposalTimeout);
    assert(provider.cancel_calls == 1);
    assert(!backend.take_submission(1U).has_value());
}

void test_capture_handoff_freezes_actual_pose_after_entering_annulus() {
    FakeProposalProvider provider;
    LearnedSmartPickupBackend backend(provider, {});
    const auto target = lifecycle_target();
    interaction::PickAssistStart start{};
    start.target_snapshot = target;
    start.affordance_id = 7U;
    start.root_world = {
        vec3(0.0F, 0.0F, 1.2F),
        quat_from_angle_axis(PIf, vec3(0.0F, 1.0F, 0.0F)),
    };
    assert(backend.begin(start, &target));

    const float facing = std::atan2(-0.2F, -0.8F);
    const interaction::Transform crossed{
        vec3(0.2F, 0.0F, 0.8F),
        quat(1.0F, 0.0F, 0.0F, 0.0F),
    };
    const interaction::PickAssistOutput handoff_turn =
        backend.observe(lifecycle_observation(10U, target, crossed));
    assert(handoff_turn.override_steering);
    assert(handoff_turn.force_strafe);
    assert(std::abs(handoff_turn.left_stick.x) < 1.0e-6F);
    assert(std::abs(handoff_turn.left_stick.z) < 1.0e-6F);
    interaction::Transform turned = crossed;
    turned.rotation = quat_from_angle_axis(
        facing, vec3(0.0F, 1.0F, 0.0F));
    backend.observe(lifecycle_observation(11U, target, turned));
    backend.observe(lifecycle_observation(12U, target, turned));
    backend.observe(lifecycle_observation(13U, target, turned));

    assert(provider.begin_calls == 1);
    assert(std::abs(provider.request.condition[18] - 0.2F) < 1.0e-6F);
    assert(std::abs(provider.request.condition[19] - 0.8F) < 1.0e-6F);
}

void test_provider_proposal_terminal_is_anchored_to_object_not_entry() {
    FakeProposalProvider provider;
    provider.ready = true;
    LearnedSmartPickupBackend backend(provider, {});
    auto target = lifecycle_target();
    target.object_world.position = vec3(2.0F, 0.8F, -1.0F);
    const interaction::Transform entry{
        vec3(2.0F, 0.0F, 0.0F),
        quat_from_angle_axis(PIf, vec3(0.0F, 1.0F, 0.0F)),
    };
    interaction::PickAssistStart start{};
    start.target_snapshot = target;
    start.affordance_id = 7U;
    start.root_world = entry;
    assert(backend.begin(start, &target));
    backend.observe(lifecycle_observation(10U, target, entry));
    backend.observe(lifecycle_observation(11U, target, entry));
    backend.observe(lifecycle_observation(12U, target, entry));
    const auto selection = backend.observe(
        lifecycle_observation(13U, target, entry));

    assert(selection.preview_requests.size() == 31U);
    for (const auto& request : selection.preview_requests) {
        assert(std::abs(request.root.world_x - 2.18F) < 1.0e-5F);
        assert(std::abs(request.root.world_z - (-0.06F)) < 1.0e-5F);
        assert(std::abs(std::abs(request.root.world_yaw_radians) - PIf) <
               1.0e-5F);
    }
}

void test_no_accepted_proposal_and_malformed_preview_fail_closed() {
    const auto target = lifecycle_target();
    const auto entry = lifecycle_entry();
    {
        FakeProposalProvider provider;
        provider.ready = true;
        provider.any_accepted = false;
        LearnedSmartPickupBackend backend(provider, {});
        begin_and_settle(backend, target, entry);
        backend.observe(lifecycle_observation(13U, target, entry));
        assert(backend.learned_diagnostics().failure_reason ==
               interaction::LearnedPickupFailureReason::NoAcceptedProposal);
    }
    {
        FakeProposalProvider provider;
        provider.ready = true;
        LearnedSmartPickupBackend backend(provider, {});
        begin_and_settle(backend, target, entry);
        const auto selection =
            backend.observe(lifecycle_observation(13U, target, entry));
        assert(!selection.preview_requests.empty());
        backend.observe(lifecycle_observation(14U, target, entry));
        assert(backend.learned_diagnostics().failure_reason ==
               interaction::LearnedPickupFailureReason::SelectionPreviewRejected);
    }
}

void test_target_mutation_and_cancel_fail_closed() {
    const auto target = lifecycle_target();
    const auto entry = lifecycle_entry();
    FakeProposalProvider provider;
    LearnedSmartPickupBackend backend(provider, {});
    interaction::PickAssistStart start{};
    start.target_snapshot = target;
    start.affordance_id = 7U;
    start.root_world = entry;
    assert(backend.begin(start, &target));
    auto changed = target;
    ++changed.handle.generation;
    backend.observe(lifecycle_observation(0U, changed, entry));
    assert(backend.learned_diagnostics().failure_reason ==
           interaction::LearnedPickupFailureReason::TargetChanged);

    FakeProposalProvider cancelled_provider;
    LearnedSmartPickupBackend cancelled(cancelled_provider, {});
    assert(cancelled.begin(start, &target));
    cancelled.cancel();
    assert(cancelled.learned_diagnostics().failure_reason ==
           interaction::LearnedPickupFailureReason::Cancelled);
    assert(!cancelled.active());
}

}  // namespace

int main() {
    test_reports_accepted_proposal_count();
    test_delegates_begin_failure_to_authored_controller();
    test_arm_selects_first_accepted_seed();
    test_follower_mirrors_spatial_lookahead_and_completion();
    test_follow_before_arm_publishes_nothing();
    test_missed_tick_cancellation_is_mirrored();
    test_backend_cancel_terminates_the_learned_follower();
    test_arm_is_rejected_when_no_proposal_accepted();
    test_provider_backed_lifecycle_submits_after_native_funnel();
    test_provider_timeout_fails_closed_and_cancels_worker();
    test_capture_handoff_freezes_actual_pose_after_entering_annulus();
    test_provider_proposal_terminal_is_anchored_to_object_not_entry();
    test_no_accepted_proposal_and_malformed_preview_fail_closed();
    test_target_mutation_and_cancel_fail_closed();
    return 0;
}
