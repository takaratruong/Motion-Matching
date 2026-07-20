#pragma once

// Strict, portable learned-funnel proposal artifact.
//
// An asynchronous worker publishes a complete canonical little-endian binary
// blob. The controller only loads the completed response and strictly matches
// it to the frozen request before learned movement starts.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace interaction {

// Frozen artifact dimensions. Exactly one condition vector, 32 proposals, 16
// samples per proposal, and four float32 values per sample. The byte and field
// order is frozen as object-local (x, z, sin(yaw), cos(yaw)).
constexpr int kFunnelConditionDim = 24;
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
    float z = 0.0F;
    float yaw_sin = 0.0F;
    float yaw_cos = 0.0F;
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

    uint64_t request_id() const { return request_id_; }
    uint64_t batch_seed() const { return batch_seed_; }
    const std::array<uint8_t, 32>& checkpoint_sha256() const {
        return checkpoint_sha256_;
    }
    const std::array<float, kFunnelConditionDim>& condition() const {
        return condition_;
    }
    const std::array<FunnelProposal, kFunnelProposalCount>& proposals() const {
        return proposals_;
    }

private:
    InteractionFunnelArtifact() = default;

    uint64_t request_id_ = 0U;
    uint64_t batch_seed_ = 0U;
    std::array<uint8_t, 32> checkpoint_sha256_{};
    std::array<float, kFunnelConditionDim> condition_{};
    std::array<FunnelProposal, kFunnelProposalCount> proposals_{};
};

}  // namespace interaction
