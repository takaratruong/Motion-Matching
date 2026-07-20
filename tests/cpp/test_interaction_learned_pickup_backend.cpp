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
using interaction::kFunnelProposalCount;
using interaction::kFunnelSampleCount;
using interaction::kFunnelSampleWidth;

// The learned backend must be usable wherever the authored backend is.
static_assert(std::is_base_of_v<SmartPickupAssistBackend, LearnedSmartPickupBackend>);
static_assert(std::has_virtual_destructor_v<SmartPickupAssistBackend>);

constexpr char kMagic[8] = {'G', '1', 'F', 'U', 'N', 'N', 'L', '1'};

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

struct SampleTuple {
    float x, y, cos, sin;
};

// Build a valid artifact where proposal 0 is rejected-and-discontinuous and the
// rest are accepted smooth funnels advancing 0.04 m per sample along +x.
InteractionFunnelArtifact make_artifact() {
    std::vector<uint8_t> bytes;
    for (char c : kMagic) bytes.push_back(static_cast<uint8_t>(c));
    append_u32(bytes, 1U);
    append_u32(bytes, static_cast<uint32_t>(kFunnelConditionDim));
    append_u32(bytes, static_cast<uint32_t>(kFunnelProposalCount));
    append_u32(bytes, static_cast<uint32_t>(kFunnelSampleCount));
    append_u32(bytes, static_cast<uint32_t>(kFunnelSampleWidth));
    for (int i = 0; i < kFunnelConditionDim; ++i) append_f32(bytes, 0.0F);
    for (int p = 0; p < kFunnelProposalCount; ++p) {
        for (int s = 0; s < kFunnelSampleCount; ++s) {
            SampleTuple sample{0.04F * static_cast<float>(s), 0.0F, 1.0F, 0.0F};
            if (p == 0 && s == 8) sample.x = 5.0F;  // discontinuous, only ok if rejected
            append_f32(bytes, sample.x);
            append_f32(bytes, sample.y);
            append_f32(bytes, sample.cos);
            append_f32(bytes, sample.sin);
        }
    }
    for (int p = 0; p < kFunnelProposalCount; ++p) append_u64(bytes, static_cast<uint64_t>(100 + p));
    for (int p = 0; p < kFunnelProposalCount; ++p) bytes.push_back(p == 0 ? 0U : 1U);
    return InteractionFunnelArtifact::load(bytes);
}

interaction::FunnelSample sample_at(float x) {
    interaction::FunnelSample s{};
    s.x = x;
    s.y = 0.0F;
    s.yaw_cos = 1.0F;
    s.yaw_sin = 0.0F;
    return s;
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

void test_follower_publishes_accepted_samples_once() {
    LearnedSmartPickupBackend backend(make_artifact());
    assert(backend.arm(0U));
    interaction::FunnelFollowerOutput first = backend.follow({0U, sample_at(0.0F)});
    assert(first.published);
    assert(first.sample.x == 0.0F);
    for (int s = 1; s < kFunnelSampleCount; ++s) {
        interaction::FunnelFollowerOutput out =
            backend.follow({static_cast<uint64_t>(s), sample_at(0.04F * static_cast<float>(s - 1))});
        assert(out.published);
        assert(out.sample.x == 0.04F * static_cast<float>(s));
    }
    assert(backend.learned_diagnostics().follower_state == FunnelFollowerState::Completed);
    assert(backend.learned_diagnostics().follower_published_count == kFunnelSampleCount);
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

void test_arm_is_rejected_when_no_proposal_accepted() {
    // Build an artifact with every proposal rejected.
    std::vector<uint8_t> bytes;
    for (char c : kMagic) bytes.push_back(static_cast<uint8_t>(c));
    append_u32(bytes, 1U);
    append_u32(bytes, static_cast<uint32_t>(kFunnelConditionDim));
    append_u32(bytes, static_cast<uint32_t>(kFunnelProposalCount));
    append_u32(bytes, static_cast<uint32_t>(kFunnelSampleCount));
    append_u32(bytes, static_cast<uint32_t>(kFunnelSampleWidth));
    for (int i = 0; i < kFunnelConditionDim; ++i) append_f32(bytes, 0.0F);
    for (int p = 0; p < kFunnelProposalCount; ++p) {
        for (int s = 0; s < kFunnelSampleCount; ++s) {
            append_f32(bytes, 0.04F * static_cast<float>(s));
            append_f32(bytes, 0.0F);
            append_f32(bytes, 1.0F);
            append_f32(bytes, 0.0F);
        }
    }
    for (int p = 0; p < kFunnelProposalCount; ++p) append_u64(bytes, static_cast<uint64_t>(p));
    for (int p = 0; p < kFunnelProposalCount; ++p) bytes.push_back(0U);
    LearnedSmartPickupBackend backend(InteractionFunnelArtifact::load(bytes));
    assert(backend.learned_diagnostics().accepted_proposal_count == 0);
    assert(!backend.arm(0U));
    assert(!backend.learned_diagnostics().armed);
}

}  // namespace

int main() {
    test_reports_accepted_proposal_count();
    test_delegates_begin_failure_to_authored_controller();
    test_arm_selects_first_accepted_seed();
    test_follower_publishes_accepted_samples_once();
    test_follow_before_arm_publishes_nothing();
    test_missed_tick_cancellation_is_mirrored();
    test_arm_is_rejected_when_no_proposal_accepted();
    return 0;
}
