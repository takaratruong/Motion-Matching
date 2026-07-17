#pragma once

#include "database.h"

#include <cfloat>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace stationary_motion_matching {

struct CandidateConfig {
    int horizon_frames = 25;
    float maximum_planar_speed_mps = 0.10F;
    float maximum_planar_displacement_m = 0.05F;
};

struct SearchResult {
    int frame = -1;
    float cost = FLT_MAX;
};

namespace detail {

inline bool finite_bits(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

inline bool finite(vec3 value) {
    return finite_bits(value.x) && finite_bits(value.y) &&
           finite_bits(value.z);
}

inline void validate_config(const CandidateConfig& config) {
    if (config.horizon_frames <= 0 ||
        !finite_bits(config.maximum_planar_speed_mps) ||
        config.maximum_planar_speed_mps < 0.0F ||
        !finite_bits(config.maximum_planar_displacement_m) ||
        config.maximum_planar_displacement_m < 0.0F) {
        throw std::invalid_argument("invalid stationary candidate config");
    }
}

inline void validate_candidate_database(const database& db) {
    const int frames = db.bone_positions.rows;
    if (frames <= 0 || db.bone_positions.cols <= 0 ||
        db.bone_positions.data == nullptr ||
        db.bone_velocities.rows != frames ||
        db.bone_velocities.cols != db.bone_positions.cols ||
        db.bone_velocities.data == nullptr ||
        db.contact_states.rows != frames || db.contact_states.cols < 2 ||
        db.contact_states.data == nullptr || db.range_starts.size <= 0 ||
        db.range_starts.size != db.range_stops.size ||
        db.range_starts.data == nullptr || db.range_stops.data == nullptr) {
        throw std::invalid_argument("malformed stationary candidate database");
    }

    int previous_stop = -1;
    for (int range = 0; range < db.range_starts.size; ++range) {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        if (start < 0 || start >= stop || stop > frames ||
            (range > 0 && start < previous_stop)) {
            throw std::invalid_argument("malformed stationary clip ranges");
        }
        previous_stop = stop;
    }
}

inline int containing_clip_stop(const database& db, int frame) {
    for (int range = 0; range < db.range_starts.size; ++range) {
        if (frame >= db.range_starts(range) &&
            frame < db.range_stops(range)) {
            return db.range_stops(range);
        }
    }
    return -1;
}

inline bool is_candidate_validated(
    const database& db,
    int frame,
    const CandidateConfig& config) {
    const int clip_stop = containing_clip_stop(db, frame);
    if (clip_stop < 0) return false;

    const int available_frames = clip_stop - 1 - frame;
    if (config.horizon_frames > available_frames) return false;
    const int window_stop = frame + config.horizon_frames;

    const double maximum_speed_squared =
        static_cast<double>(config.maximum_planar_speed_mps) *
        config.maximum_planar_speed_mps;
    bool eligible = true;
    for (int sample = frame; sample <= window_stop; ++sample) {
        const vec3 position = db.bone_positions(sample, 0);
        const vec3 velocity = db.bone_velocities(sample, 0);
        if (!finite(position) || !finite(velocity)) {
            throw std::invalid_argument("nonfinite stationary root sample");
        }
        const double x = velocity.x;
        const double z = velocity.z;
        if (x * x + z * z > maximum_speed_squared ||
            db.contact_states(sample, 0) != true ||
            db.contact_states(sample, 1) != true) {
            eligible = false;
        }
    }

    const vec3 start = db.bone_positions(frame, 0);
    const vec3 stop = db.bone_positions(window_stop, 0);
    const double displacement_x =
        static_cast<double>(stop.x) - static_cast<double>(start.x);
    const double displacement_z =
        static_cast<double>(stop.z) - static_cast<double>(start.z);
    const double maximum_displacement_squared =
        static_cast<double>(config.maximum_planar_displacement_m) *
        config.maximum_planar_displacement_m;
    return eligible &&
           displacement_x * displacement_x +
                   displacement_z * displacement_z <=
               maximum_displacement_squared;
}

inline void validate_search_shape(
    const database& db,
    const slice1d<float> query,
    const std::vector<int>& candidates) {
    const int frames = db.nframes();
    const int feature_width = db.features.cols;
    if (frames <= 0 || db.features.rows != frames || feature_width <= 0 ||
        db.features.data == nullptr || query.size != feature_width ||
        query.data == nullptr || db.features_offset.size != feature_width ||
        db.features_offset.data == nullptr ||
        db.features_scale.size != feature_width ||
        db.features_scale.data == nullptr || candidates.empty()) {
        throw std::invalid_argument("malformed stationary search shape");
    }

    int previous = -1;
    for (size_t index = 0; index < candidates.size(); ++index) {
        const int candidate = candidates[index];
        if (candidate < 0 || candidate >= frames ||
            (index > 0 && candidate <= previous)) {
            throw std::invalid_argument("invalid stationary candidate index");
        }
        previous = candidate;
    }
}

}  // namespace detail

inline bool is_candidate(
    const database& db,
    int frame,
    const CandidateConfig& config = {}) {
    detail::validate_config(config);
    detail::validate_candidate_database(db);
    if (frame < 0 || frame >= db.nframes()) {
        throw std::invalid_argument("stationary candidate frame out of bounds");
    }
    return detail::is_candidate_validated(db, frame, config);
}

inline std::vector<int> derive_candidates(
    const database& db,
    const CandidateConfig& config = {}) {
    detail::validate_config(config);
    detail::validate_candidate_database(db);

    std::vector<int> candidates;
    for (int frame = 0; frame < db.nframes(); ++frame) {
        if (detail::is_candidate_validated(db, frame, config)) {
            candidates.push_back(frame);
        }
    }
    return candidates;
}

inline SearchResult search(
    const database& db,
    const slice1d<float> query,
    const std::vector<int>& candidates) {
    detail::validate_search_shape(db, query, candidates);

    std::vector<float> normalized(static_cast<size_t>(db.nfeatures()));
    for (int dimension = 0; dimension < db.nfeatures(); ++dimension) {
        const float offset = db.features_offset(dimension);
        const float scale = db.features_scale(dimension);
        if (!detail::finite_bits(offset) || !detail::finite_bits(scale) ||
            !(scale > 0.0F) || !detail::finite_bits(query(dimension))) {
            throw std::invalid_argument("invalid stationary search feature");
        }
        normalized[static_cast<size_t>(dimension)] =
            (query(dimension) - offset) / scale;
        if (!detail::finite_bits(
                normalized[static_cast<size_t>(dimension)])) {
            throw std::invalid_argument("nonfinite normalized search query");
        }
    }

    SearchResult result;
    for (int candidate : candidates) {
        float cost = 0.0F;
        for (int dimension = 0; dimension < db.nfeatures(); ++dimension) {
            const float feature = db.features(candidate, dimension);
            if (!detail::finite_bits(feature)) {
                throw std::invalid_argument(
                    "nonfinite stationary candidate feature");
            }
            const float difference =
                normalized[static_cast<size_t>(dimension)] - feature;
            if (!detail::finite_bits(difference)) {
                throw std::invalid_argument("nonfinite stationary difference");
            }
            cost += difference * difference;
            if (!detail::finite_bits(cost)) {
                throw std::invalid_argument("nonfinite stationary search cost");
            }
        }
        if (result.frame == -1 || cost < result.cost) {
            result.frame = candidate;
            result.cost = cost;
        }
    }
    return result;
}

}  // namespace stationary_motion_matching
