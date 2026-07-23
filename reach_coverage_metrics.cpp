#include "reach_coverage_metrics.h"

#include "g1_skeleton.h"
#include "reach_placement.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace reach {
namespace {

constexpr float kTwoPi = 6.28318530717958647692F;

bool finite(vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

}  // namespace

size_t count_placed_root_azimuth_sectors(
    const Pack& pack,
    const std::vector<Candidate>& candidates,
    vec3 target_position,
    size_t sector_count) {
    if (!finite(target_position) || sector_count == 0U) {
        throw std::invalid_argument("invalid root azimuth sector query");
    }
    std::vector<bool> occupied(sector_count, false);
    for (const Candidate& candidate : candidates) {
        if (candidate.clip >= pack.database.clip_count) {
            throw std::out_of_range(
                "root azimuth candidate clip outside database");
        }
        const int32_t terminal_frame =
            pack.database.range_stops.at(candidate.clip) - 1;
        const interaction::WorldPose terminal = interaction::world_pose(
            place_pose(
                pack,
                candidate.clip,
                candidate.yaw_index,
                target_position,
                terminal_frame));
        vec3 direction =
            terminal.positions[g1_skeleton::Simulation] - target_position;
        direction.y = 0.0F;
        if (length(direction) <= 1.0e-5F) continue;
        float angle = std::atan2(direction.z, direction.x);
        if (angle < 0.0F) angle += kTwoPi;
        const size_t sector = std::min(
            static_cast<size_t>(
                angle * static_cast<float>(sector_count) / kTwoPi),
            sector_count - 1U);
        occupied[sector] = true;
    }
    return static_cast<size_t>(
        std::count(occupied.begin(), occupied.end(), true));
}

uint64_t accepted_candidate_set_hash(
    const std::vector<Candidate>& candidates) {
    std::vector<std::pair<size_t, uint8_t>> identities;
    identities.reserve(candidates.size());
    for (const Candidate& candidate : candidates) {
        identities.emplace_back(candidate.clip, candidate.yaw_index);
    }
    std::sort(identities.begin(), identities.end());

    constexpr uint64_t offset = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset;
    const auto mix = [&](uint8_t byte) {
        hash ^= static_cast<uint64_t>(byte);
        hash *= prime;
    };
    for (const auto& identity : identities) {
        const uint64_t clip = static_cast<uint64_t>(identity.first);
        for (unsigned shift = 0U; shift < 64U; shift += 8U) {
            mix(static_cast<uint8_t>((clip >> shift) & 0xffU));
        }
        mix(identity.second);
    }
    return hash;
}

}  // namespace reach
