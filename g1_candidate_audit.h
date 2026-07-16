#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "array.h"

static constexpr int G1CandidateAuditFeatureCount = 31;
static constexpr int G1CandidateAuditTopCapacity = 16;
static constexpr int G1CandidateAuditIgnoreRangeEnd = 20;
static constexpr int G1CandidateAuditIgnoreSurrounding = 20;
static constexpr uint32_t G1CandidateAuditDelta0_25Bits =
    UINT32_C(0x3e800000);
static constexpr uint32_t G1CandidateAuditDelta1Bits =
    UINT32_C(0x3f800000);
static constexpr uint32_t G1CandidateAuditDelta4Bits =
    UINT32_C(0x40800000);

struct G1CandidateAuditEntry
{
    int frame = -1;
    int range = -1;
    uint32_t cost_bits = 0;
};

struct G1CandidateAudit
{
    uint32_t eligible_count = 0;
    uint32_t within_best_plus_0_25 = 0;
    uint32_t within_best_plus_1 = 0;
    uint32_t within_best_plus_4 = 0;
    uint32_t top_count = 0;
    G1CandidateAuditEntry top[G1CandidateAuditTopCapacity];
};

static inline uint32_t g1_candidate_audit_float_bits(float value)
{
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static inline float g1_candidate_audit_float_from_bits(uint32_t value)
{
    float output = 0.0f;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static inline bool g1_candidate_audit_float_is_finite(float value)
{
    return (g1_candidate_audit_float_bits(value) & UINT32_C(0x7f800000)) !=
           UINT32_C(0x7f800000);
}

static inline bool g1_candidate_audit_cost_bits_are_valid(uint32_t value)
{
    return (value & UINT32_C(0x80000000)) == 0 &&
           (value & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

#if defined(__GNUC__) && !defined(__clang__)
#define G1_CANDIDATE_AUDIT_STRICT_ARITHMETIC \
    __attribute__((noinline, unused, optimize("no-fast-math")))
#else
#define G1_CANDIDATE_AUDIT_STRICT_ARITHMETIC
#endif

static G1_CANDIDATE_AUDIT_STRICT_ARITHMETIC bool
g1_candidate_audit_subtract(float& output, float left, float right)
{
    volatile float rounded = left - right;
    output = rounded;
    return g1_candidate_audit_float_is_finite(output);
}

static G1_CANDIDATE_AUDIT_STRICT_ARITHMETIC bool
g1_candidate_audit_multiply(float& output, float left, float right)
{
    volatile float rounded = left * right;
    output = rounded;
    return g1_candidate_audit_float_is_finite(output);
}

static G1_CANDIDATE_AUDIT_STRICT_ARITHMETIC bool
g1_candidate_audit_add(float& output, float left, float right)
{
    volatile float rounded = left + right;
    output = rounded;
    return g1_candidate_audit_float_is_finite(output);
}

#undef G1_CANDIDATE_AUDIT_STRICT_ARITHMETIC

static inline bool g1_candidate_audit_checked_bytes(
    std::size_t& output,
    int count,
    std::size_t element_size)
{
    output = 0;
    if (count < 0 || element_size == 0) return false;
    const std::size_t converted = static_cast<std::size_t>(count);
    if (converted >
        std::numeric_limits<std::size_t>::max() / element_size) {
        return false;
    }
    output = converted * element_size;
    return true;
}

static inline bool g1_candidate_audit_checked_matrix_bytes(
    std::size_t& output,
    int rows,
    int columns,
    std::size_t element_size)
{
    output = 0;
    if (rows < 0 || columns < 0 || element_size == 0) return false;
    const std::size_t converted_rows = static_cast<std::size_t>(rows);
    const std::size_t converted_columns =
        static_cast<std::size_t>(columns);
    if (converted_columns != 0 &&
        converted_rows >
            std::numeric_limits<std::size_t>::max() / converted_columns) {
        return false;
    }
    const std::size_t elements = converted_rows * converted_columns;
    if (elements >
        std::numeric_limits<std::size_t>::max() / element_size) {
        return false;
    }
    output = elements * element_size;
    return true;
}

static inline bool g1_candidate_audit_span(
    uintptr_t& begin,
    uintptr_t& end,
    const void* memory,
    std::size_t size)
{
    begin = 0;
    end = 0;
    if (size == 0) return true;
    if (memory == NULL) return false;
    begin = reinterpret_cast<uintptr_t>(memory);
    if (begin > std::numeric_limits<uintptr_t>::max() - size) return false;
    end = begin + size;
    return true;
}

static inline bool g1_candidate_audit_overlaps(
    const void* first,
    std::size_t first_size,
    const void* second,
    std::size_t second_size)
{
    if (first_size == 0 || second_size == 0) return false;
    uintptr_t first_begin = 0;
    uintptr_t first_end = 0;
    uintptr_t second_begin = 0;
    uintptr_t second_end = 0;
    if (!g1_candidate_audit_span(
            first_begin, first_end, first, first_size) ||
        !g1_candidate_audit_span(
            second_begin, second_end, second, second_size)) {
        return true;
    }
    return first_begin < second_end && second_begin < first_end;
}

static inline bool g1_candidate_audit_increment(uint32_t& value)
{
    if (value == std::numeric_limits<uint32_t>::max()) return false;
    ++value;
    return true;
}

static inline bool g1_candidate_audit_entry_precedes(
    float left_cost,
    int left_frame,
    float right_cost,
    int right_frame)
{
    if (left_cost < right_cost) return true;
    if (right_cost < left_cost) return false;
    return left_frame < right_frame;
}

static inline bool g1_candidate_audit_is_valid(
    const G1CandidateAudit& value)
{
    if (value.within_best_plus_0_25 > value.within_best_plus_1 ||
        value.within_best_plus_1 > value.within_best_plus_4 ||
        value.within_best_plus_4 > value.eligible_count) {
        return false;
    }
    const uint32_t expected_top = value.eligible_count <
            static_cast<uint32_t>(G1CandidateAuditTopCapacity)
        ? value.eligible_count
        : static_cast<uint32_t>(G1CandidateAuditTopCapacity);
    if (value.top_count != expected_top) return false;
    if (value.eligible_count > 0 &&
        (value.within_best_plus_0_25 == 0 ||
         value.within_best_plus_1 == 0 ||
         value.within_best_plus_4 == 0)) {
        return false;
    }

    for (uint32_t index = 0; index < value.top_count; ++index) {
        const G1CandidateAuditEntry& entry = value.top[index];
        if (entry.frame < 0 || entry.range < 0 ||
            !g1_candidate_audit_cost_bits_are_valid(entry.cost_bits)) {
            return false;
        }
        for (uint32_t previous = 0; previous < index; ++previous) {
            if (value.top[previous].frame == entry.frame) return false;
        }
        if (index > 0) {
            const G1CandidateAuditEntry& previous = value.top[index - 1];
            const float previous_cost =
                g1_candidate_audit_float_from_bits(previous.cost_bits);
            const float current_cost =
                g1_candidate_audit_float_from_bits(entry.cost_bits);
            if (g1_candidate_audit_entry_precedes(
                    current_cost,
                    entry.frame,
                    previous_cost,
                    previous.frame)) {
                return false;
            }
        }
    }
    for (uint32_t index = value.top_count;
         index < static_cast<uint32_t>(G1CandidateAuditTopCapacity);
         ++index) {
        if (value.top[index].frame != -1 ||
            value.top[index].range != -1 ||
            value.top[index].cost_bits != 0) {
            return false;
        }
    }
    return true;
}

static inline bool g1_candidate_audit_frame_cost(
    float& output,
    const float* query,
    const float* feature_row)
{
    float cost = 0.0f;
    for (int feature = 0;
         feature < G1CandidateAuditFeatureCount;
         ++feature) {
        float difference = 0.0f;
        float square = 0.0f;
        float next = 0.0f;
        if (!g1_candidate_audit_subtract(
                difference, query[feature], feature_row[feature]) ||
            !g1_candidate_audit_multiply(
                square, difference, difference) ||
            !g1_candidate_audit_add(next, cost, square)) {
            return false;
        }
        cost = next;
    }
    output = cost;
    return true;
}

static inline bool g1_candidate_audit_build(
    G1CandidateAudit& output,
    const slice1d<float> query_normalized,
    const slice2d<float> features,
    const slice1d<int> range_starts,
    const slice1d<int> range_stops,
    int incumbent_index,
    int ignore_range_end = G1CandidateAuditIgnoreRangeEnd,
    int ignore_surrounding = G1CandidateAuditIgnoreSurrounding)
{
    if (query_normalized.size != G1CandidateAuditFeatureCount ||
        query_normalized.data == NULL ||
        features.rows <= 0 ||
        features.cols != G1CandidateAuditFeatureCount ||
        features.data == NULL ||
        range_starts.size <= 0 ||
        range_starts.size != range_stops.size ||
        range_starts.data == NULL ||
        range_stops.data == NULL ||
        (incumbent_index != -1 &&
         (incumbent_index < 0 || incumbent_index >= features.rows)) ||
        ignore_range_end != G1CandidateAuditIgnoreRangeEnd ||
        ignore_surrounding != G1CandidateAuditIgnoreSurrounding) {
        return false;
    }

    std::size_t query_bytes = 0;
    std::size_t feature_bytes = 0;
    std::size_t range_bytes = 0;
    if (!g1_candidate_audit_checked_bytes(
            query_bytes, query_normalized.size, sizeof(float)) ||
        !g1_candidate_audit_checked_matrix_bytes(
            feature_bytes, features.rows, features.cols, sizeof(float)) ||
        !g1_candidate_audit_checked_bytes(
            range_bytes, range_starts.size, sizeof(int))) {
        return false;
    }
    uintptr_t unused_begin = 0;
    uintptr_t unused_end = 0;
    if (!g1_candidate_audit_span(
            unused_begin, unused_end, query_normalized.data, query_bytes) ||
        !g1_candidate_audit_span(
            unused_begin, unused_end, features.data, feature_bytes) ||
        !g1_candidate_audit_span(
            unused_begin, unused_end, range_starts.data, range_bytes) ||
        !g1_candidate_audit_span(
            unused_begin, unused_end, range_stops.data, range_bytes) ||
        g1_candidate_audit_overlaps(
            &output, sizeof(output), query_normalized.data, query_bytes) ||
        g1_candidate_audit_overlaps(
            &output, sizeof(output), features.data, feature_bytes) ||
        g1_candidate_audit_overlaps(
            &output, sizeof(output), range_starts.data, range_bytes) ||
        g1_candidate_audit_overlaps(
            &output, sizeof(output), range_stops.data, range_bytes)) {
        return false;
    }

    int cursor = 0;
    for (int range = 0; range < range_starts.size; ++range) {
        const int start = range_starts(range);
        const int stop = range_stops(range);
        if (start != cursor || stop <= start || stop > features.rows) {
            return false;
        }
        cursor = stop;
    }
    if (cursor != features.rows) return false;

    for (int feature = 0;
         feature < G1CandidateAuditFeatureCount;
         ++feature) {
        if (!g1_candidate_audit_float_is_finite(
                query_normalized(feature))) {
            return false;
        }
    }
    const std::size_t feature_count = feature_bytes / sizeof(float);
    for (std::size_t index = 0; index < feature_count; ++index) {
        if (!g1_candidate_audit_float_is_finite(features.data[index])) {
            return false;
        }
    }

    G1CandidateAudit candidate;
    float top_costs[G1CandidateAuditTopCapacity] = {};
    for (int range = 0; range < range_starts.size; ++range) {
        const int start = range_starts(range);
        const int length = range_stops(range) - start;
        const int eligible_length = length > ignore_range_end
            ? length - ignore_range_end
            : 0;
        const int search_stop = start + eligible_length;
        for (int frame = start; frame < search_stop; ++frame) {
            const int64_t separation = incumbent_index < 0
                ? std::numeric_limits<int64_t>::max()
                : (frame >= incumbent_index
                       ? static_cast<int64_t>(frame) - incumbent_index
                       : static_cast<int64_t>(incumbent_index) - frame);
            if (incumbent_index >= 0 &&
                separation < ignore_surrounding) {
                continue;
            }
            if (!g1_candidate_audit_increment(candidate.eligible_count)) {
                return false;
            }

            float cost = 0.0f;
            const float* const feature_row =
                features.data +
                static_cast<std::size_t>(frame) *
                    G1CandidateAuditFeatureCount;
            if (!g1_candidate_audit_frame_cost(
                    cost, query_normalized.data, feature_row)) {
                return false;
            }

            uint32_t insertion = candidate.top_count;
            for (uint32_t index = 0;
                 index < candidate.top_count;
                 ++index) {
                if (g1_candidate_audit_entry_precedes(
                        cost, frame, top_costs[index],
                        candidate.top[index].frame)) {
                    insertion = index;
                    break;
                }
            }
            if (insertion <
                    static_cast<uint32_t>(G1CandidateAuditTopCapacity) ||
                candidate.top_count <
                    static_cast<uint32_t>(G1CandidateAuditTopCapacity)) {
                const uint32_t next_count = candidate.top_count <
                        static_cast<uint32_t>(G1CandidateAuditTopCapacity)
                    ? candidate.top_count + 1U
                    : candidate.top_count;
                for (uint32_t index = next_count;
                     index > insertion + 1U;
                     --index) {
                    candidate.top[index - 1U] = candidate.top[index - 2U];
                    top_costs[index - 1U] = top_costs[index - 2U];
                }
                if (insertion < next_count) {
                    candidate.top[insertion].frame = frame;
                    candidate.top[insertion].range = range;
                    candidate.top[insertion].cost_bits =
                        g1_candidate_audit_float_bits(cost);
                    top_costs[insertion] = cost;
                }
                candidate.top_count = next_count;
            }
        }
    }

    if (candidate.eligible_count > 0) {
        float threshold_0_25 = 0.0f;
        float threshold_1 = 0.0f;
        float threshold_4 = 0.0f;
        const float best = top_costs[0];
        if (!g1_candidate_audit_add(
                threshold_0_25,
                best,
                g1_candidate_audit_float_from_bits(
                    G1CandidateAuditDelta0_25Bits)) ||
            !g1_candidate_audit_add(
                threshold_1,
                best,
                g1_candidate_audit_float_from_bits(
                    G1CandidateAuditDelta1Bits)) ||
            !g1_candidate_audit_add(
                threshold_4,
                best,
                g1_candidate_audit_float_from_bits(
                    G1CandidateAuditDelta4Bits))) {
            return false;
        }

        for (int range = 0; range < range_starts.size; ++range) {
            const int start = range_starts(range);
            const int length = range_stops(range) - start;
            const int eligible_length = length > ignore_range_end
                ? length - ignore_range_end
                : 0;
            const int search_stop = start + eligible_length;
            for (int frame = start; frame < search_stop; ++frame) {
                const int64_t separation = incumbent_index < 0
                    ? std::numeric_limits<int64_t>::max()
                    : (frame >= incumbent_index
                           ? static_cast<int64_t>(frame) - incumbent_index
                           : static_cast<int64_t>(incumbent_index) - frame);
                if (incumbent_index >= 0 &&
                    separation < ignore_surrounding) {
                    continue;
                }
                float cost = 0.0f;
                const float* const feature_row =
                    features.data +
                    static_cast<std::size_t>(frame) *
                        G1CandidateAuditFeatureCount;
                if (!g1_candidate_audit_frame_cost(
                        cost, query_normalized.data, feature_row)) {
                    return false;
                }
                if (cost <= threshold_0_25 &&
                    !g1_candidate_audit_increment(
                        candidate.within_best_plus_0_25)) {
                    return false;
                }
                if (cost <= threshold_1 &&
                    !g1_candidate_audit_increment(
                        candidate.within_best_plus_1)) {
                    return false;
                }
                if (cost <= threshold_4 &&
                    !g1_candidate_audit_increment(
                        candidate.within_best_plus_4)) {
                    return false;
                }
            }
        }
    }

    if (!g1_candidate_audit_is_valid(candidate)) return false;
    output = candidate;
    return true;
}
