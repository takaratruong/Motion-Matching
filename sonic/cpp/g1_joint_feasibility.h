#pragma once

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "database.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include "g1_joint_projection.h"
#include "sha256.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

struct sonic_joint_feasibility_certificate
{
    array1d<unsigned char> raw_safe;
    array1d<unsigned char> search_safe;
    int frame_count = 0;
    int raw_safe_count = 0;
    int raw_unsafe_count = 0;
    int search_safe_count = 0;
    int joint_limit_violation_count[SonicG1JointCount] = {};
    char mask_sha256[65] = {};
};

static inline bool sonic_joint_feasibility_error(
    char* output,
    int capacity,
    const char* message)
{
    if (output != nullptr && capacity > 0) {
        std::snprintf(
            output,
            static_cast<std::size_t>(capacity),
            "%s",
            message);
    }
    return false;
}

static inline bool sonic_joint_feasibility_projection_error(
    char* output,
    int capacity,
    int frame,
    const char* failure,
    const char* detail)
{
    if (output != nullptr && capacity > 0) {
        std::snprintf(
            output,
            static_cast<std::size_t>(capacity),
            "frame %d projection %s failure: %s",
            frame,
            failure,
            detail != nullptr && detail[0] != '\0' ? detail : "no detail");
    }
    return false;
}

static inline const char* sonic_joint_projection_failure_name(
    sonic_joint_projection_failure failure)
{
    switch (failure) {
    case SonicJointProjectionValid: return "valid";
    case SonicJointProjectionShape: return "shape";
    case SonicJointProjectionContract: return "contract";
    case SonicJointProjectionInput: return "input";
    case SonicJointProjectionSingular: return "singular";
    case SonicJointProjectionResidual: return "residual";
    case SonicJointProjectionLimit: return "limit";
    case SonicJointProjectionVelocity: return "velocity";
    }
    return "invalid-diagnostic";
}

static inline bool sonic_joint_feasibility_digest(
    char (&output)[65],
    int frame_count,
    const array1d<unsigned char>& raw_safe,
    const array1d<unsigned char>& search_safe,
    char* error,
    int error_capacity)
{
    if (frame_count < 0) {
        return sonic_joint_feasibility_error(
            error,
            error_capacity,
            "joint feasibility certificate has negative frame count");
    }
    if (raw_safe.size != frame_count || search_safe.size != frame_count ||
        (frame_count > 0 &&
         (raw_safe.data == nullptr || search_safe.data == nullptr))) {
        return sonic_joint_feasibility_error(
            error,
            error_capacity,
            "joint feasibility certificate mask shape mismatch");
    }

    static constexpr char schema[] =
        "g1-joint-feasibility-certificate/v1";
    sha256_state state;
    sha256_update(
        state,
        reinterpret_cast<const std::uint8_t*>(schema),
        sizeof(schema) - 1U);

    std::uint8_t encoded_frame_count[8] = {};
    const std::uint64_t unsigned_frame_count =
        static_cast<std::uint64_t>(frame_count);
    for (int byte = 0; byte < 8; ++byte) {
        encoded_frame_count[byte] = static_cast<std::uint8_t>(
            unsigned_frame_count >> static_cast<unsigned int>(byte * 8));
    }
    sha256_update(state, encoded_frame_count, sizeof(encoded_frame_count));
    sha256_update(
        state,
        reinterpret_cast<const std::uint8_t*>(raw_safe.data),
        static_cast<std::size_t>(frame_count));
    sha256_update(
        state,
        reinterpret_cast<const std::uint8_t*>(search_safe.data),
        static_cast<std::size_t>(frame_count));

    const std::string digest = sha256_finish(state);
    if (digest.size() != 64U) {
        return sonic_joint_feasibility_error(
            error,
            error_capacity,
            "joint feasibility certificate SHA-256 length mismatch");
    }
    std::memcpy(output, digest.data(), digest.size());
    output[64] = '\0';
    return true;
}

static inline bool sonic_build_joint_feasibility_certificate(
    sonic_joint_feasibility_certificate& out,
    const database& db,
    const sonic_joint_contract_entry (&contract)[SonicG1JointCount],
    char* error,
    int error_capacity)
{
    if (error != nullptr && error_capacity > 0) error[0] = '\0';

    const int frame_count = db.nframes();
    if (frame_count < 0) {
        return sonic_joint_feasibility_error(
            error,
            error_capacity,
            "joint feasibility certificate has negative frame count");
    }
    if (frame_count == 0) {
        return sonic_joint_feasibility_error(
            error,
            error_capacity,
            "joint feasibility certificate frame 0 must be raw-safe");
    }
    if (db.bone_rotations.rows != frame_count ||
        db.bone_angular_velocities.rows != frame_count ||
        db.bone_rotations.data == nullptr ||
        db.bone_angular_velocities.data == nullptr) {
        return sonic_joint_feasibility_projection_error(
            error,
            error_capacity,
            0,
            "shape",
            "database rotation/angular-velocity row shape mismatch");
    }

    sonic_joint_feasibility_certificate candidate;
    candidate.frame_count = frame_count;
    candidate.raw_safe.resize(frame_count);
    candidate.raw_safe.zero();
    candidate.search_safe.resize(frame_count);
    candidate.search_safe.zero();

    for (int frame = 0; frame < frame_count; ++frame) {
        float positions[SonicG1JointCount];
        float velocities[SonicG1JointCount];
        float residuals[SonicG1JointCount];
        sonic_joint_projection_diagnostic diagnostic;
        char projection_error[512] = {};
        const bool projected = sonic_project_joint_state(
            positions,
            velocities,
            residuals,
            diagnostic,
            contract,
            db.bone_rotations(frame),
            db.bone_angular_velocities(frame),
            projection_error,
            static_cast<int>(sizeof(projection_error)));

        if (projected) {
            if (diagnostic.failure != SonicJointProjectionValid ||
                diagnostic.row != -1) {
                return sonic_joint_feasibility_projection_error(
                    error,
                    error_capacity,
                    frame,
                    "invalid-diagnostic",
                    "successful projection returned a failure diagnostic");
            }
            for (int row = 0; row < SonicG1JointCount; ++row) {
                if (!std::isfinite(positions[row]) ||
                    !std::isfinite(velocities[row]) ||
                    !std::isfinite(residuals[row])) {
                    return sonic_joint_feasibility_projection_error(
                        error,
                        error_capacity,
                        frame,
                        "non-finite",
                        "successful projection returned non-finite output");
                }
            }
            candidate.raw_safe(frame) = 1U;
            ++candidate.raw_safe_count;
            continue;
        }

        if (diagnostic.failure == SonicJointProjectionLimit &&
            diagnostic.row >= 0 &&
            diagnostic.row < SonicG1JointCount &&
            std::isfinite(diagnostic.position) &&
            std::isfinite(diagnostic.lower) &&
            std::isfinite(diagnostic.upper) &&
            diagnostic.lower <= diagnostic.upper) {
            candidate.raw_safe(frame) = 0U;
            ++candidate.raw_unsafe_count;
            ++candidate.joint_limit_violation_count[diagnostic.row];
            continue;
        }

        const char* failure = sonic_joint_projection_failure_name(
            diagnostic.failure);
        if (diagnostic.failure == SonicJointProjectionLimit) {
            failure = "invalid-diagnostic";
        }
        return sonic_joint_feasibility_projection_error(
            error,
            error_capacity,
            frame,
            failure,
            projection_error);
    }

    if (candidate.raw_safe(0) != 1U) {
        return sonic_joint_feasibility_error(
            error,
            error_capacity,
            "joint feasibility certificate frame 0 must be raw-safe");
    }
    if (db.range_starts.size <= 0 ||
        db.range_stops.size != db.range_starts.size ||
        db.range_starts.data == nullptr || db.range_stops.data == nullptr) {
        return sonic_joint_feasibility_error(
            error,
            error_capacity,
            "joint feasibility certificate range shape mismatch");
    }

    int expected_start = 0;
    for (int range = 0; range < db.nranges(); ++range) {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        if (start != expected_start || stop <= start || stop > frame_count) {
            return sonic_joint_feasibility_error(
                error,
                error_capacity,
                "joint feasibility certificate range is not contiguous/in-bounds");
        }
        for (int frame = start; frame < stop; ++frame) {
            const int successor = frame + 1 < stop ? frame + 1 : frame;
            candidate.search_safe(frame) = static_cast<unsigned char>(
                candidate.raw_safe(frame) == 1U &&
                candidate.raw_safe(successor) == 1U);
            candidate.search_safe_count += candidate.search_safe(frame);
        }
        expected_start = stop;
    }
    if (expected_start != frame_count) {
        return sonic_joint_feasibility_error(
            error,
            error_capacity,
            "joint feasibility certificate ranges do not cover all frames");
    }

    int violation_count = 0;
    for (int row = 0; row < SonicG1JointCount; ++row) {
        if (candidate.joint_limit_violation_count[row] < 0) {
            return sonic_joint_feasibility_error(
                error,
                error_capacity,
                "joint feasibility certificate has negative violation count");
        }
        violation_count += candidate.joint_limit_violation_count[row];
    }
    int verified_raw_safe_count = 0;
    int verified_search_safe_count = 0;
    for (int frame = 0; frame < frame_count; ++frame) {
        if (candidate.raw_safe(frame) > 1U ||
            candidate.search_safe(frame) > 1U) {
            return sonic_joint_feasibility_error(
                error,
                error_capacity,
                "joint feasibility certificate mask is not binary");
        }
        verified_raw_safe_count += candidate.raw_safe(frame);
        verified_search_safe_count += candidate.search_safe(frame);
    }
    if (candidate.raw_safe_count + candidate.raw_unsafe_count !=
            frame_count ||
        candidate.raw_safe_count != verified_raw_safe_count ||
        candidate.search_safe_count != verified_search_safe_count ||
        candidate.raw_unsafe_count != violation_count) {
        return sonic_joint_feasibility_error(
            error,
            error_capacity,
            "joint feasibility certificate counts do not reconcile");
    }
    if (candidate.search_safe_count <= 0) {
        return sonic_joint_feasibility_error(
            error,
            error_capacity,
            "joint feasibility certificate has no search-safe frames");
    }
    if (!sonic_joint_feasibility_digest(
            candidate.mask_sha256,
            candidate.frame_count,
            candidate.raw_safe,
            candidate.search_safe,
            error,
            error_capacity)) {
        return false;
    }

    out = candidate;
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return true;
}
