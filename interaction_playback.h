#pragma once

#include "interaction_matcher.h"

#include <cstdint>

namespace interaction {

class SequentialPlayer {
public:
    explicit SequentialPlayer(const Database& database);
    void start(
        const MatchCandidate& candidate,
        const Pose& current,
        float speed = 1.0F);
    void advance(float dt);
    Pose sample() const;
    Phase phase() const;
    int32_t frame() const;
    bool at_contact() const;
    bool at_hold() const;
    bool finished() const;
    float elapsed_seconds() const;
    vec3 entry_root_correction() const;

private:
    const Database* database_ = nullptr;
    MatchCandidate candidate_{};
    float source_frame_ = 0.0F;
    float elapsed_ = 0.0F;
    float speed_ = 1.0F;
    double source_frame_exact_ = 0.0;
    double elapsed_exact_ = 0.0;
    int32_t final_frame_ = -1;
    bool started_ = false;
};

}  // namespace interaction
