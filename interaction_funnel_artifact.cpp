#include "interaction_funnel_artifact.h"

#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>

namespace interaction {

namespace {

// Canonical little-endian header magic. "G1FUNNL1".
constexpr char kMagic[8] = {'G', '1', 'F', 'U', 'N', 'N', 'L', '1'};
constexpr uint32_t kSchemaVersion = 1U;

// Yaw unit-length tolerance; mirrors the Python writer's np.allclose atol.
constexpr float kYawUnitTolerance = 2.0e-5F;

// Sequential little-endian reader that never reads past the buffer end.
class Cursor {
public:
    Cursor(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    uint32_t read_u32() {
        require(4U);
        uint32_t value = 0U;
        for (int i = 0; i < 4; ++i) {
            value |= static_cast<uint32_t>(data_[offset_ + static_cast<size_t>(i)])
                     << (8 * i);
        }
        offset_ += 4U;
        return value;
    }

    uint64_t read_u64() {
        require(8U);
        uint64_t value = 0U;
        for (int i = 0; i < 8; ++i) {
            value |= static_cast<uint64_t>(data_[offset_ + static_cast<size_t>(i)])
                     << (8 * i);
        }
        offset_ += 8U;
        return value;
    }

    float read_f32() {
        uint32_t raw = read_u32();
        float value = 0.0F;
        std::memcpy(&value, &raw, sizeof(value));
        return value;
    }

    uint8_t read_u8() {
        require(1U);
        return data_[offset_++];
    }

    void expect_magic() {
        require(sizeof(kMagic));
        if (std::memcmp(data_ + offset_, kMagic, sizeof(kMagic)) != 0) {
            throw std::runtime_error("funnel artifact: bad magic");
        }
        offset_ += sizeof(kMagic);
    }

    void expect_end() const {
        if (offset_ != size_) {
            throw std::runtime_error("funnel artifact: trailing bytes");
        }
    }

private:
    void require(size_t count) const {
        if (offset_ + count > size_) {
            throw std::runtime_error("funnel artifact: truncated buffer");
        }
    }

    const uint8_t* data_;
    size_t size_;
    size_t offset_ = 0U;
};

float read_finite(Cursor& cursor, const char* what) {
    float value = cursor.read_f32();
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string("funnel artifact: non-finite ") + what);
    }
    return value;
}

// Signed yaw step between two consecutive unit vectors (cos, sin).
float yaw_step(const FunnelSample& previous, const FunnelSample& current) {
    float cross = current.yaw_cos * previous.yaw_sin - current.yaw_sin * previous.yaw_cos;
    float dot = current.yaw_cos * previous.yaw_cos + current.yaw_sin * previous.yaw_sin;
    return std::abs(std::atan2(cross, dot));
}

void certify_accepted(const FunnelProposal& proposal) {
    float arc = 0.0F;
    for (int s = 1; s < kFunnelSampleCount; ++s) {
        const FunnelSample& previous = proposal.samples[static_cast<size_t>(s - 1)];
        const FunnelSample& current = proposal.samples[static_cast<size_t>(s)];
        float dx = current.x - previous.x;
        float dy = current.y - previous.y;
        float step = std::sqrt(dx * dx + dy * dy);
        arc += step;
        if (step > kFunnelMaxTranslationStep) {
            throw std::runtime_error(
                "funnel artifact: accepted proposal violates translation step");
        }
        if (yaw_step(previous, current) > kFunnelMaxYawStepRadians) {
            throw std::runtime_error(
                "funnel artifact: accepted proposal violates yaw step");
        }
    }
    if (arc < kFunnelMinArcLength) {
        throw std::runtime_error(
            "funnel artifact: accepted proposal violates minimum arc length");
    }
}

}  // namespace

InteractionFunnelArtifact InteractionFunnelArtifact::load(
    const std::vector<uint8_t>& bytes) {
    Cursor cursor(bytes.data(), bytes.size());
    cursor.expect_magic();

    if (cursor.read_u32() != kSchemaVersion) {
        throw std::runtime_error("funnel artifact: unsupported schema version");
    }
    if (cursor.read_u32() != static_cast<uint32_t>(kFunnelConditionDim) ||
        cursor.read_u32() != static_cast<uint32_t>(kFunnelProposalCount) ||
        cursor.read_u32() != static_cast<uint32_t>(kFunnelSampleCount) ||
        cursor.read_u32() != static_cast<uint32_t>(kFunnelSampleWidth)) {
        throw std::runtime_error("funnel artifact: invalid dimensions");
    }

    InteractionFunnelArtifact artifact;
    for (int i = 0; i < kFunnelConditionDim; ++i) {
        artifact.condition_[static_cast<size_t>(i)] = read_finite(cursor, "condition");
    }

    for (int p = 0; p < kFunnelProposalCount; ++p) {
        FunnelProposal& proposal = artifact.proposals_[static_cast<size_t>(p)];
        for (int s = 0; s < kFunnelSampleCount; ++s) {
            FunnelSample& sample = proposal.samples[static_cast<size_t>(s)];
            sample.x = read_finite(cursor, "sample");
            sample.y = read_finite(cursor, "sample");
            sample.yaw_cos = read_finite(cursor, "sample");
            sample.yaw_sin = read_finite(cursor, "sample");
            float norm = std::sqrt(
                sample.yaw_cos * sample.yaw_cos + sample.yaw_sin * sample.yaw_sin);
            if (std::abs(norm - 1.0F) > kYawUnitTolerance) {
                throw std::runtime_error("funnel artifact: non-unit yaw vector");
            }
        }
    }

    for (int p = 0; p < kFunnelProposalCount; ++p) {
        artifact.proposals_[static_cast<size_t>(p)].seed = cursor.read_u64();
    }
    for (int p = 0; p < kFunnelProposalCount; ++p) {
        uint8_t flag = cursor.read_u8();
        if (flag > 1U) {
            throw std::runtime_error("funnel artifact: invalid accepted flag");
        }
        artifact.proposals_[static_cast<size_t>(p)].accepted = (flag != 0U);
    }

    cursor.expect_end();

    std::set<uint64_t> seen_seeds;
    for (const FunnelProposal& proposal : artifact.proposals_) {
        if (!seen_seeds.insert(proposal.seed).second) {
            throw std::runtime_error("funnel artifact: duplicate seed");
        }
        if (proposal.accepted) {
            certify_accepted(proposal);
        }
    }

    return artifact;
}

}  // namespace interaction
