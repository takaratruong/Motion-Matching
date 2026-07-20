#include "interaction_funnel_artifact.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using interaction::InteractionFunnelArtifact;
using interaction::kFunnelConditionDim;
using interaction::kFunnelProposalCount;
using interaction::kFunnelSampleCount;
using interaction::kFunnelSampleWidth;

// Mirror the on-disk canonical little-endian layout so the test can build
// deliberately corrupt buffers without depending on the loader internals.
constexpr char kMagic[8] = {'G', '1', 'F', 'U', 'N', 'N', 'L', '1'};

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFFU));
    }
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFFU));
    }
}

void append_f32(std::vector<uint8_t>& bytes, float value) {
    uint32_t raw = 0U;
    std::memcpy(&raw, &value, sizeof(raw));
    append_u32(bytes, raw);
}

struct BufferPlan {
    std::array<float, kFunnelConditionDim> condition{};
    // proposals[p][s][component]
    std::array<std::array<std::array<float, kFunnelSampleWidth>, kFunnelSampleCount>,
               kFunnelProposalCount>
        proposals{};
    std::array<uint64_t, kFunnelProposalCount> seeds{};
    std::array<uint8_t, kFunnelProposalCount> accepted{};
};

// A smooth, certifiable proposal: advances 0.2 m along x over 16 samples with a
// constant unit yaw vector (0, 1). translation step ~= 0.0133 m, arc = 0.2 m.
BufferPlan make_valid_plan() {
    BufferPlan plan{};
    for (int p = 0; p < kFunnelProposalCount; ++p) {
        for (int s = 0; s < kFunnelSampleCount; ++s) {
            float t = static_cast<float>(s) / static_cast<float>(kFunnelSampleCount - 1);
            plan.proposals[p][s][0] = 0.2F * t;  // x
            plan.proposals[p][s][1] = 0.0F;       // y
            plan.proposals[p][s][2] = 0.0F;       // yaw cos
            plan.proposals[p][s][3] = 1.0F;       // yaw sin
        }
        plan.seeds[p] = static_cast<uint64_t>(p);
        plan.accepted[p] = 1U;
    }
    return plan;
}

std::vector<uint8_t> serialize(const BufferPlan& plan) {
    std::vector<uint8_t> bytes;
    for (char c : kMagic) {
        bytes.push_back(static_cast<uint8_t>(c));
    }
    append_u32(bytes, 1U);  // schema_version
    append_u32(bytes, static_cast<uint32_t>(kFunnelConditionDim));
    append_u32(bytes, static_cast<uint32_t>(kFunnelProposalCount));
    append_u32(bytes, static_cast<uint32_t>(kFunnelSampleCount));
    append_u32(bytes, static_cast<uint32_t>(kFunnelSampleWidth));
    for (float value : plan.condition) {
        append_f32(bytes, value);
    }
    for (const auto& proposal : plan.proposals) {
        for (const auto& sample : proposal) {
            for (float value : sample) {
                append_f32(bytes, value);
            }
        }
    }
    for (uint64_t seed : plan.seeds) {
        append_u64(bytes, seed);
    }
    for (uint8_t flag : plan.accepted) {
        bytes.push_back(flag);
    }
    return bytes;
}

bool throws_runtime(const std::vector<uint8_t>& bytes) {
    try {
        InteractionFunnelArtifact::load(bytes);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void test_valid_round_trip_preserves_all_fields() {
    BufferPlan plan = make_valid_plan();
    for (int i = 0; i < kFunnelConditionDim; ++i) {
        plan.condition[i] = static_cast<float>(i) * 0.5F;
    }
    plan.accepted[5] = 0U;  // a non-accepted proposal is still preserved

    InteractionFunnelArtifact artifact = InteractionFunnelArtifact::load(serialize(plan));

    for (int i = 0; i < kFunnelConditionDim; ++i) {
        assert(artifact.condition()[static_cast<size_t>(i)] == static_cast<float>(i) * 0.5F);
    }
    const auto& proposals = artifact.proposals();
    for (int p = 0; p < kFunnelProposalCount; ++p) {
        assert(proposals[static_cast<size_t>(p)].seed == static_cast<uint64_t>(p));
        assert(proposals[static_cast<size_t>(p)].accepted == (p != 5));
        assert(proposals[static_cast<size_t>(p)].samples[15].x == 0.2F);
        assert(proposals[static_cast<size_t>(p)].samples[0].yaw_sin == 1.0F);
    }
}

void test_truncated_buffer_is_rejected() {
    std::vector<uint8_t> bytes = serialize(make_valid_plan());
    bytes.pop_back();
    assert(throws_runtime(bytes));
}

void test_trailing_bytes_are_rejected() {
    std::vector<uint8_t> bytes = serialize(make_valid_plan());
    bytes.push_back(0U);
    assert(throws_runtime(bytes));
}

void test_bad_magic_is_rejected() {
    std::vector<uint8_t> bytes = serialize(make_valid_plan());
    bytes[0] = 'X';
    assert(throws_runtime(bytes));
}

void test_bad_schema_version_is_rejected() {
    std::vector<uint8_t> bytes = serialize(make_valid_plan());
    bytes[8] = 2U;  // first byte of schema_version
    assert(throws_runtime(bytes));
}

void test_wrong_dimension_header_is_rejected() {
    BufferPlan plan = make_valid_plan();
    std::vector<uint8_t> bytes = serialize(plan);
    // proposal_count field lives at offset 16; corrupt low byte 32 -> 31.
    bytes[16] = 31U;
    assert(throws_runtime(bytes));
}

void test_non_finite_float_is_rejected() {
    BufferPlan plan = make_valid_plan();
    plan.condition[3] = std::numeric_limits<float>::quiet_NaN();
    assert(throws_runtime(serialize(plan)));

    BufferPlan plan_inf = make_valid_plan();
    plan_inf.proposals[2][4][0] = std::numeric_limits<float>::infinity();
    assert(throws_runtime(serialize(plan_inf)));
}

void test_non_unit_yaw_vector_is_rejected() {
    BufferPlan plan = make_valid_plan();
    plan.proposals[7][3][2] = 2.0F;  // (2, 1) is not unit length
    assert(throws_runtime(serialize(plan)));
}

void test_duplicate_seeds_are_rejected() {
    BufferPlan plan = make_valid_plan();
    plan.seeds[9] = plan.seeds[8];
    assert(throws_runtime(serialize(plan)));
}

void test_accepted_translation_violation_is_rejected() {
    BufferPlan plan = make_valid_plan();
    plan.proposals[1][8][0] = 1.0F;  // large jump exceeds 0.08 m step
    assert(plan.accepted[1] == 1U);
    assert(throws_runtime(serialize(plan)));
}

void test_accepted_yaw_violation_is_rejected() {
    BufferPlan plan = make_valid_plan();
    // Flip yaw vector by 90 degrees between consecutive samples (still unit).
    plan.proposals[3][5][2] = 1.0F;
    plan.proposals[3][5][3] = 0.0F;
    assert(throws_runtime(serialize(plan)));
}

void test_accepted_arc_violation_is_rejected() {
    BufferPlan plan = make_valid_plan();
    // Collapse proposal 4 to a stationary point: arc = 0 < 0.15 m.
    for (int s = 0; s < kFunnelSampleCount; ++s) {
        plan.proposals[4][s][0] = 0.0F;
    }
    assert(throws_runtime(serialize(plan)));
}

void test_rejected_proposal_may_violate_continuity() {
    BufferPlan plan = make_valid_plan();
    // A discontinuous proposal that is NOT accepted must still load.
    plan.proposals[6][8][0] = 5.0F;
    plan.accepted[6] = 0U;
    InteractionFunnelArtifact artifact = InteractionFunnelArtifact::load(serialize(plan));
    assert(artifact.proposals()[6].accepted == false);
    assert(artifact.proposals()[6].samples[8].x == 5.0F);
}

}  // namespace

int main() {
    test_valid_round_trip_preserves_all_fields();
    test_truncated_buffer_is_rejected();
    test_trailing_bytes_are_rejected();
    test_bad_magic_is_rejected();
    test_bad_schema_version_is_rejected();
    test_wrong_dimension_header_is_rejected();
    test_non_finite_float_is_rejected();
    test_non_unit_yaw_vector_is_rejected();
    test_duplicate_seeds_are_rejected();
    test_accepted_translation_violation_is_rejected();
    test_accepted_yaw_violation_is_rejected();
    test_accepted_arc_violation_is_rejected();
    test_rejected_proposal_may_violate_continuity();
    return 0;
}
