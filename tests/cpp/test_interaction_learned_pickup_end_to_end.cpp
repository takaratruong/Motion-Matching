#include "interaction_learned_pickup_backend.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <vector>

namespace {

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (int index = 0; index < 4; ++index) {
        bytes.push_back(static_cast<uint8_t>((value >> (8 * index)) & 0xffU));
    }
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (int index = 0; index < 8; ++index) {
        bytes.push_back(static_cast<uint8_t>((value >> (8 * index)) & 0xffU));
    }
}

void append_f32(std::vector<uint8_t>& bytes, float value) {
    uint32_t raw = 0U;
    std::memcpy(&raw, &value, sizeof(raw));
    append_u32(bytes, raw);
}

interaction::InteractionFunnelArtifact artifact_for(
    const interaction::FunnelProposalRequest& request) {
    const bool alternate = request.condition[18] < -0.001F;
    std::vector<uint8_t> bytes;
    for (char value : {'G', '1', 'F', 'U', 'N', 'N', 'L', '3'}) {
        bytes.push_back(static_cast<uint8_t>(value));
    }
    append_u32(bytes, 3U);
    append_u32(bytes, 24U);
    append_u32(bytes, 32U);
    append_u32(bytes, 16U);
    append_u32(bytes, 4U);
    append_u64(bytes, request.request_id);
    append_u64(bytes, request.batch_seed);
    bytes.insert(bytes.end(), request.checkpoint_sha256.begin(),
                 request.checkpoint_sha256.end());
    for (float value : request.condition) append_f32(bytes, value);
    for (int proposal = 0; proposal < 32; ++proposal) {
        for (int knot = 0; knot < 16; ++knot) {
            const float alpha = static_cast<float>(knot) / 15.0F;
            const float bow = (alternate ? 0.02F : -0.02F) *
                std::sin(PIf * alpha);
            float x = request.condition[18] +
                alpha * (0.18F - request.condition[18]) + bow;
            float z = request.condition[19] +
                alpha * (-0.94F - request.condition[19]);
            float yaw_sin = request.condition[20];
            float yaw_cos = request.condition[21];
            if (knot == 0) {
                x = request.condition[18];
                z = request.condition[19];
                yaw_sin = request.condition[20];
                yaw_cos = request.condition[21];
            }
            append_f32(bytes, x);
            append_f32(bytes, z);
            append_f32(bytes, yaw_sin);
            append_f32(bytes, yaw_cos);
        }
    }
    for (int proposal = 0; proposal < 32; ++proposal) {
        append_u64(bytes, static_cast<uint64_t>(100 + proposal));
    }
    for (int proposal = 0; proposal < 32; ++proposal) bytes.push_back(1U);
    return interaction::InteractionFunnelArtifact::load(bytes);
}

class ImmediateProvider final : public interaction::FunnelProposalProvider {
public:
    bool begin(const interaction::FunnelProposalRequest& request) override {
        request_ = request;
        active_ = true;
        ++begin_count;
        return true;
    }

    interaction::FunnelProposalPoll poll() override {
        if (!active_) return {};
        active_ = false;
        return {
            interaction::FunnelProposalPollState::Ready,
            artifact_for(request_),
            {},
        };
    }

    void cancel() override { active_ = false; }

    int begin_count = 0;
    interaction::FunnelProposalRequest request_{};
    bool active_ = false;
};

interaction::PickAssistObservation observation(
    uint64_t tick,
    const interaction::InteractionTarget& target,
    interaction::Transform root) {
    interaction::PickAssistObservation value{};
    value.controller_tick = tick;
    value.runtime_state = interaction::RuntimeState::Locomotion;
    value.target = &target;
    value.displayed_root = root;
    value.snapshot_fingerprint = tick + 500U;
    return value;
}

interaction::FunnelSample object_local_to_world_sample(
    const interaction::Transform& object,
    const interaction::FunnelSample& local) {
    const vec3 forward = quat_mul_vec3(
        object.rotation, vec3(0.0F, 0.0F, 1.0F));
    const float object_yaw = std::atan2(forward.x, forward.z);
    const float sine = std::sin(object_yaw);
    const float cosine = std::cos(object_yaw);
    const float local_yaw = std::atan2(local.yaw_sin, local.yaw_cos);
    const float world_yaw = object_yaw + local_yaw;
    return {
        object.position.x + cosine * local.x - sine * local.z,
        object.position.z + sine * local.x + cosine * local.z,
        std::sin(world_yaw),
        std::cos(world_yaw),
    };
}

interaction::RuntimeInput idle_runtime_input(
    const interaction::LocomotionSnapshot& locomotion) {
    interaction::RuntimeInput input{};
    input.dt = 1.0F / 25.0F;
    input.locomotion = locomotion;
    return input;
}

struct TrialEvidence {
    std::array<float, 24> condition{};
    uint64_t proposal_seed = 0U;
    int proposal_index = -1;
    std::array<interaction::FunnelSample,
               interaction::kFunnelExecutionTickCount> world_targets{};
    int attachment_edges = 0;
};

TrialEvidence run_learned_funnel_to_actual_carry(float bearing_radians) {
    interaction::RuntimeFixture fixture = interaction::make_runtime_fixture();
    interaction::InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry,
        interaction::RuntimeConfig{});
    const interaction::InteractionTarget* target =
        fixture.registry.find(fixture.request.target);
    assert(target != nullptr);
    assert(target->affordances.front().interaction_slots.empty());

    ImmediateProvider provider;
    interaction::LearnedSmartPickupBackend backend(provider, {});
    interaction::Transform entry{
        fixture.locomotion.pose.positions[g1_skeleton::Simulation],
        quat_from_angle_axis(
            bearing_radians, vec3(0.0F, 1.0F, 0.0F)),
    };
    entry.position.x = -std::sin(bearing_radians);
    entry.position.z = 3.0F - std::cos(bearing_radians);
    interaction::PickAssistStart start{};
    start.target_snapshot = *target;
    start.affordance_id = fixture.request.affordance_id;
    start.root_world = entry;
    assert(backend.begin(start, target));
    const interaction::PickAssistOutput selection =
        backend.observe(observation(0U, *target, entry));
    assert(provider.begin_count == 1);
    assert(selection.preview_requests.size() == 32U);

    interaction::PickAssistObservation selected =
        observation(1U, *target, entry);
    selected.preview_snapshot_fingerprint = selected.snapshot_fingerprint;
    for (const auto& request : selection.preview_requests) {
        const std::optional<interaction::PickEntryPreview> preview =
            runtime.preview_pick(
                fixture.locomotion, request.root,
                target->handle, fixture.request.affordance_id);
        assert(preview.has_value());
        assert(preview->path_feasible);
        assert(preview->match_ready);
        selected.preview_results.push_back({
            request,
            preview,
        });
    }
    backend.observe(selected);
    assert(backend.learned_diagnostics().state ==
           interaction::LearnedPickupState::FunnelFollow);

    TrialEvidence evidence{};
    evidence.condition = backend.learned_diagnostics().frozen_condition;
    evidence.proposal_seed = backend.learned_diagnostics().armed_seed;
    evidence.proposal_index =
        backend.learned_diagnostics().selected_proposal_index;
    std::array<interaction::FunnelSample, 16> knots{};
    const bool alternate =
        backend.learned_diagnostics().frozen_condition[18] < -0.001F;
    for (int knot = 0; knot < 16; ++knot) {
        const float alpha = static_cast<float>(knot) / 15.0F;
        const float bow = (alternate ? 0.02F : -0.02F) *
            std::sin(PIf * alpha);
        knots[static_cast<size_t>(knot)] = {
            evidence.condition[18] +
                alpha * (0.18F - evidence.condition[18]) + bow,
            evidence.condition[19] +
                alpha * (-0.94F - evidence.condition[19]),
            evidence.condition[20],
            evidence.condition[21],
        };
    }
    knots.front() = {
        evidence.condition[18], evidence.condition[19],
        evidence.condition[20], evidence.condition[21],
    };
    const auto targets = interaction::expand_funnel_execution(knots);
    for (size_t tick = 0U; tick < targets.size(); ++tick) {
        evidence.world_targets[tick] = object_local_to_world_sample(
            target->object_world, targets[tick]);
    }
    const interaction::FunnelSample terminal_world =
        object_local_to_world_sample(target->object_world, targets.back());
    interaction::Transform tracked{
        vec3(terminal_world.x, entry.position.y, terminal_world.z),
        quat_from_angle_axis(
            std::atan2(terminal_world.yaw_sin, terminal_world.yaw_cos),
            vec3(0.0F, 1.0F, 0.0F)),
    };
    backend.observe(observation(2U, *target, tracked));
    backend.observe(observation(3U, *target, tracked));
    const interaction::PickAssistOutput final_request = backend.observe(
        observation(4U, *target, tracked));
    assert(backend.learned_diagnostics().follower_progress_index ==
           interaction::kFunnelExecutionTickCount - 1);
    assert(backend.learned_diagnostics().follower_lookahead_index ==
           interaction::kFunnelExecutionTickCount - 1);
    assert(final_request.preview_requests.size() == 1U);

    interaction::PickAssistObservation final =
        observation(5U, *target, tracked);
    final.preview_snapshot_fingerprint = final.snapshot_fingerprint;
    const auto& request = final_request.preview_requests.front();
    final.preview_results.push_back({
        request,
        runtime.preview_pick(
            fixture.locomotion, request.root,
            target->handle, fixture.request.affordance_id),
    });
    assert(backend.observe(final).submit_interact);
    const std::optional<interaction::PickRequest> learned_request =
        backend.take_submission(fixture.request.request_id);
    assert(learned_request.has_value());
    assert(learned_request->target == fixture.request.target);
    assert(learned_request->affordance_id == fixture.request.affordance_id);

    interaction::RuntimeInput activation =
        idle_runtime_input(fixture.locomotion);
    activation.interact_pressed = true;
    activation.pick_request = learned_request;
    interaction::RuntimeOutput output = runtime.update(activation);
    bool was_attached = false;
    int attachment_edges = 0;
    for (int update = 0;
         update < 200 && output.diagnostics.state !=
             interaction::RuntimeState::Carry;
         ++update) {
        if (output.diagnostics.attached && !was_attached) ++attachment_edges;
        was_attached = output.diagnostics.attached;
        output = runtime.update(idle_runtime_input(fixture.locomotion));
    }
    if (output.diagnostics.attached && !was_attached) ++attachment_edges;
    assert(output.diagnostics.state == interaction::RuntimeState::Carry);
    assert(output.diagnostics.attached);
    assert(attachment_edges == 1);
    const interaction::InteractionTarget* held =
        fixture.registry.find(fixture.request.target);
    assert(held != nullptr);
    assert(held->state == interaction::ObjectState::Held);
    assert(held->owner_request == fixture.request.request_id);
    const interaction::RuntimeOutput carry = runtime.update(
        idle_runtime_input(fixture.locomotion));
    assert(carry.diagnostics.state == interaction::RuntimeState::Carry);
    assert(carry.diagnostics.attached);
    assert(backend.learned_diagnostics().selected_proposal_index >= 0);
    evidence.attachment_edges = attachment_edges;
    return evidence;
}

}  // namespace

int main() {
    const TrialEvidence first = run_learned_funnel_to_actual_carry(0.0F);
    const TrialEvidence second = run_learned_funnel_to_actual_carry(0.04F);
    assert(std::memcmp(
               first.condition.data(), second.condition.data(),
               sizeof(first.condition)) != 0);
    assert(std::abs(first.world_targets.back().x -
                    second.world_targets.back().x) < 1.0e-6F);
    assert(std::abs(first.world_targets.back().z -
                    second.world_targets.back().z) < 1.0e-6F);
    assert(first.world_targets[20].x != second.world_targets[20].x ||
           first.world_targets[20].z != second.world_targets[20].z);

    const std::filesystem::path directory =
        "build/g1-funnels/evidence";
    std::filesystem::create_directories(directory);
    std::ofstream output(directory / "two-bearing-headless.json");
    assert(output);
    output << std::setprecision(9) << "{\n  \"trials\": [\n";
    const TrialEvidence trials[]{first, second};
    for (size_t trial = 0U; trial < 2U; ++trial) {
        output << "    {\"condition\":[";
        for (size_t index = 0U; index < trials[trial].condition.size(); ++index) {
            if (index != 0U) output << ',';
            output << trials[trial].condition[index];
        }
        output << "],\"proposal_seed\":" << trials[trial].proposal_seed
               << ",\"proposal_index\":" << trials[trial].proposal_index
               << ",\"attachment_edges\":" << trials[trial].attachment_edges
               << ",\"final_state\":\"Carry\",\"targets\":[";
        for (size_t tick = 0U; tick < trials[trial].world_targets.size(); ++tick) {
            if (tick != 0U) output << ',';
            const auto& target = trials[trial].world_targets[tick];
            output << '[' << target.x << ',' << target.z << ','
                   << target.yaw_sin << ',' << target.yaw_cos << ']';
        }
        output << "]}" << (trial == 0U ? "," : "") << '\n';
    }
    output << "  ]\n}\n";
    assert(output.good());
    return 0;
}
