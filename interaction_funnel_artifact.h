#pragma once

// Strict, portable learned-funnel proposal artifact.
//
// The controller loads a complete offline artifact before learned mode starts;
// it never runs Python or Torch. The artifact is a canonical little-endian
// binary blob produced by resources/g1_interaction_builder/proposal_artifact.py
// and owned here by a strict C++ loader that rejects any malformed input.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace interaction {

// Frozen artifact dimensions. Exactly one condition vector, 32 proposals, 16
// samples per proposal, and four float32 values per sample (planar x, y and a
// unit yaw vector cos, sin).
constexpr int kFunnelConditionDim = 18;
constexpr int kFunnelProposalCount = 32;
constexpr int kFunnelSampleCount = 16;
constexpr int kFunnelSampleWidth = 4;

// Accepted-proposal continuity constraints.
constexpr float kFunnelMaxTranslationStep = 0.08F;       // metres between samples
constexpr float kFunnelMaxYawStepRadians = 0.26179939F;  // 15 degrees
constexpr float kFunnelMinArcLength = 0.15F;             // metres, total path

// One execution-order sample of a funnel proposal.
struct FunnelSample {
    float x = 0.0F;
    float y = 0.0F;
    float yaw_cos = 0.0F;
    float yaw_sin = 0.0F;
};

// One proposal: an immutable seed, an acceptance flag, and its samples.
struct FunnelProposal {
    uint64_t seed = 0U;
    bool accepted = false;
    std::array<FunnelSample, kFunnelSampleCount> samples{};
};

// Immutable, fully validated artifact. Construct only via load(); a returned
// instance is guaranteed to satisfy every structural and continuity invariant.
class InteractionFunnelArtifact {
public:
    // Parse and strictly validate the canonical little-endian byte buffer.
    // Throws std::runtime_error on any malformed input: bad magic, unsupported
    // schema, wrong dimensions, truncation, trailing bytes, non-finite floats,
    // non-unit yaw vectors, duplicate seeds, or an accepted proposal that
    // violates the translation, yaw, or arc-length constraints.
    static InteractionFunnelArtifact load(const std::vector<uint8_t>& bytes);

    const std::array<float, kFunnelConditionDim>& condition() const {
        return condition_;
    }
    const std::array<FunnelProposal, kFunnelProposalCount>& proposals() const {
        return proposals_;
    }

private:
    InteractionFunnelArtifact() = default;

    std::array<float, kFunnelConditionDim> condition_{};
    std::array<FunnelProposal, kFunnelProposalCount> proposals_{};
};

}  // namespace interaction
