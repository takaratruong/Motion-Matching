#if defined(__FAST_MATH__)
#error "g1_candidate_recovery.cpp requires strict floating-point compilation"
#endif

#include "g1_candidate_recovery.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

struct G1RecoveryStrictAuditSink
{
    uint32_t normalized_query_bits[G1RecoveryFeatureCount] = {};
    uint32_t recovery_incumbent_score_bits = 0U;
    uint32_t exhaustive_rows_tested = 0U;
};

static uint32_t g1_recovery_float_bits(float value)
{
    uint32_t output = 0U;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

static bool g1_recovery_float_is_finite(float value)
{
    return (g1_recovery_float_bits(value) & UINT32_C(0x7f800000)) !=
           UINT32_C(0x7f800000);
}

static bool g1_recovery_fail(char* error, int error_capacity, const char* text)
{
    if (error != nullptr && error_capacity > 0) {
        std::snprintf(
            error,
            static_cast<std::size_t>(error_capacity),
            "%s",
            text);
    }
    return false;
}

static bool g1_recovery_checked_bytes(
    std::size_t& output,
    int count,
    std::size_t element_size)
{
    output = 0U;
    if (count < 0) return false;
    const std::size_t converted = static_cast<std::size_t>(count);
    if (converted >
        std::numeric_limits<std::size_t>::max() / element_size) {
        return false;
    }
    output = converted * element_size;
    return true;
}

static bool g1_recovery_checked_matrix_bytes(
    std::size_t& output,
    int rows,
    int columns,
    std::size_t element_size)
{
    output = 0U;
    if (rows < 0 || columns < 0) return false;
    const std::size_t converted_rows = static_cast<std::size_t>(rows);
    const std::size_t converted_columns =
        static_cast<std::size_t>(columns);
    if (converted_columns != 0U &&
        converted_rows >
            std::numeric_limits<std::size_t>::max() /
                converted_columns) {
        return false;
    }
    const std::size_t elements = converted_rows * converted_columns;
    if (elements > static_cast<std::size_t>(
                       std::numeric_limits<int>::max()) ||
        elements >
        std::numeric_limits<std::size_t>::max() / element_size) {
        return false;
    }
    output = elements * element_size;
    return true;
}

static bool g1_recovery_span(
    uintptr_t& begin,
    uintptr_t& end,
    const void* memory,
    std::size_t size)
{
    begin = 0U;
    end = 0U;
    if (size == 0U) return true;
    if (memory == nullptr) return false;
    begin = reinterpret_cast<uintptr_t>(memory);
    if (begin > std::numeric_limits<uintptr_t>::max() - size) {
        return false;
    }
    end = begin + size;
    return true;
}

static bool g1_recovery_overlaps(
    const void* first,
    std::size_t first_size,
    const void* second,
    std::size_t second_size)
{
    if (first_size == 0U || second_size == 0U) return false;
    uintptr_t first_begin = 0U;
    uintptr_t first_end = 0U;
    uintptr_t second_begin = 0U;
    uintptr_t second_end = 0U;
    if (!g1_recovery_span(
            first_begin, first_end, first, first_size) ||
        !g1_recovery_span(
            second_begin, second_end, second, second_size)) {
        return true;
    }
    return first_begin < second_end && second_begin < first_end;
}

static bool g1_recovery_increment(uint32_t& value)
{
    if (value == std::numeric_limits<uint32_t>::max()) return false;
    ++value;
    return true;
}

static bool g1_recovery_array_span_is_valid(
    const void* data,
    std::size_t bytes)
{
    uintptr_t begin = 0U;
    uintptr_t end = 0U;
    return g1_recovery_span(begin, end, data, bytes);
}

static bool g1_recovery_request_is_valid(
    const G1RecoveryRequest& request,
    char* error,
    int error_capacity)
{
    if (request.db == nullptr) {
        return g1_recovery_fail(
            error, error_capacity, "recovery database is null");
    }
    const database& db = *request.db;
    const int frames = db.bone_positions.rows;
    const int bones = db.bone_positions.cols;
    const int ranges = db.range_starts.size;
    if (frames <= 0 || bones <= 0 || ranges <= 0 || ranges > frames) {
        return g1_recovery_fail(
            error, error_capacity, "recovery database is empty");
    }
    if (db.features.rows != frames ||
        db.features.cols != static_cast<int>(G1RecoveryFeatureCount) ||
        db.features_offset.size !=
            static_cast<int>(G1RecoveryFeatureCount) ||
        db.features_scale.size !=
            static_cast<int>(G1RecoveryFeatureCount)) {
        return g1_recovery_fail(
            error, error_capacity, "recovery feature shape is invalid");
    }
    if (db.bone_velocities.rows != frames ||
        db.bone_velocities.cols != bones ||
        db.bone_rotations.rows != frames ||
        db.bone_rotations.cols != bones ||
        db.bone_angular_velocities.rows != frames ||
        db.bone_angular_velocities.cols != bones ||
        db.bone_parents.size != bones) {
        return g1_recovery_fail(
            error, error_capacity, "recovery pose shape is invalid");
    }
    if (db.terrain_features.rows != frames ||
        db.terrain_features.cols != 4 ||
        db.contact_states.rows != frames ||
        db.contact_states.cols <= 0) {
        return g1_recovery_fail(
            error, error_capacity, "recovery auxiliary shape is invalid");
    }
    if (db.range_stops.size != ranges) {
        return g1_recovery_fail(
            error, error_capacity, "recovery range shape is invalid");
    }

    const int expected_small = (frames - 1) / BOUND_SM_SIZE + 1;
    const int expected_large = (frames - 1) / BOUND_LR_SIZE + 1;
    if (db.bound_sm_min.rows != expected_small ||
        db.bound_sm_min.cols !=
            static_cast<int>(G1RecoveryFeatureCount) ||
        db.bound_sm_max.rows != expected_small ||
        db.bound_sm_max.cols !=
            static_cast<int>(G1RecoveryFeatureCount) ||
        db.bound_lr_min.rows != expected_large ||
        db.bound_lr_min.cols !=
            static_cast<int>(G1RecoveryFeatureCount) ||
        db.bound_lr_max.rows != expected_large ||
        db.bound_lr_max.cols !=
            static_cast<int>(G1RecoveryFeatureCount)) {
        return g1_recovery_fail(
            error, error_capacity, "recovery bound shape is invalid");
    }

    if (request.incumbent_frame < 0 ||
        request.incumbent_frame >= frames ||
        request.legacy_selected_frame < 0 ||
        request.legacy_selected_frame >= frames) {
        return g1_recovery_fail(
            error, error_capacity, "recovery frame is invalid");
    }
    if (!g1_recovery_float_is_finite(request.transition_cost) ||
        request.transition_cost < 0.0f ||
        !g1_recovery_float_is_finite(request.public_incumbent_cost) ||
        request.public_incumbent_cost < 0.0f ||
        request.ignore_range_end < 0 ||
        request.ignore_surrounding < 0) {
        return g1_recovery_fail(
            error, error_capacity, "recovery request scalar is invalid");
    }

    std::size_t pose_vec_bytes = 0U;
    std::size_t pose_quat_bytes = 0U;
    std::size_t parents_bytes = 0U;
    std::size_t ranges_bytes = 0U;
    std::size_t features_bytes = 0U;
    std::size_t feature_vector_bytes = 0U;
    std::size_t terrain_bytes = 0U;
    std::size_t contacts_bytes = 0U;
    std::size_t small_bound_bytes = 0U;
    std::size_t large_bound_bytes = 0U;
    if (!g1_recovery_checked_matrix_bytes(
            pose_vec_bytes, frames, bones, sizeof(vec3)) ||
        !g1_recovery_checked_matrix_bytes(
            pose_quat_bytes, frames, bones, sizeof(quat)) ||
        !g1_recovery_checked_bytes(
            parents_bytes, bones, sizeof(int)) ||
        !g1_recovery_checked_bytes(
            ranges_bytes, ranges, sizeof(int)) ||
        !g1_recovery_checked_matrix_bytes(
            features_bytes,
            frames,
            static_cast<int>(G1RecoveryFeatureCount),
            sizeof(float)) ||
        !g1_recovery_checked_bytes(
            feature_vector_bytes,
            static_cast<int>(G1RecoveryFeatureCount),
            sizeof(float)) ||
        !g1_recovery_checked_matrix_bytes(
            terrain_bytes, frames, 4, sizeof(float)) ||
        !g1_recovery_checked_matrix_bytes(
            contacts_bytes,
            frames,
            db.contact_states.cols,
            sizeof(bool)) ||
        !g1_recovery_checked_matrix_bytes(
            small_bound_bytes,
            expected_small,
            static_cast<int>(G1RecoveryFeatureCount),
            sizeof(float)) ||
        !g1_recovery_checked_matrix_bytes(
            large_bound_bytes,
            expected_large,
            static_cast<int>(G1RecoveryFeatureCount),
            sizeof(float))) {
        return g1_recovery_fail(
            error, error_capacity, "recovery database byte count overflow");
    }
    if (!g1_recovery_array_span_is_valid(
            db.bone_positions.data, pose_vec_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.bone_velocities.data, pose_vec_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.bone_rotations.data, pose_quat_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.bone_angular_velocities.data, pose_vec_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.bone_parents.data, parents_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.range_starts.data, ranges_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.range_stops.data, ranges_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.features.data, features_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.features_offset.data, feature_vector_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.features_scale.data, feature_vector_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.terrain_features.data, terrain_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.contact_states.data, contacts_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.bound_sm_min.data, small_bound_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.bound_sm_max.data, small_bound_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.bound_lr_min.data, large_bound_bytes) ||
        !g1_recovery_array_span_is_valid(
            db.bound_lr_max.data, large_bound_bytes)) {
        return g1_recovery_fail(
            error, error_capacity, "recovery database span is invalid");
    }

    int previous_stop = 0;
    bool incumbent_owned = false;
    bool legacy_owned = false;
    for (int range = 0; range < ranges; ++range) {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        if (start < previous_stop || stop <= start || stop > frames) {
            return g1_recovery_fail(
                error, error_capacity, "recovery range is invalid");
        }
        if (request.incumbent_frame >= start &&
            request.incumbent_frame < stop) {
            incumbent_owned = true;
        }
        if (request.legacy_selected_frame >= start &&
            request.legacy_selected_frame < stop) {
            legacy_owned = true;
        }
        previous_stop = stop;
    }
    if (!incumbent_owned || !legacy_owned) {
        return g1_recovery_fail(
            error, error_capacity, "recovery frame has no source range");
    }

    for (int box = 0; box < expected_small; ++box) {
        for (uint32_t feature = 0U;
             feature < G1RecoveryFeatureCount;
             ++feature) {
            const float minimum =
                db.bound_sm_min(box, static_cast<int>(feature));
            const float maximum =
                db.bound_sm_max(box, static_cast<int>(feature));
            if (!g1_recovery_float_is_finite(minimum) ||
                !g1_recovery_float_is_finite(maximum) ||
                minimum > maximum) {
                return g1_recovery_fail(
                    error,
                    error_capacity,
                    "recovery small bound is invalid");
            }
        }
    }
    for (int box = 0; box < expected_large; ++box) {
        for (uint32_t feature = 0U;
             feature < G1RecoveryFeatureCount;
             ++feature) {
            const float minimum =
                db.bound_lr_min(box, static_cast<int>(feature));
            const float maximum =
                db.bound_lr_max(box, static_cast<int>(feature));
            if (!g1_recovery_float_is_finite(minimum) ||
                !g1_recovery_float_is_finite(maximum) ||
                minimum > maximum) {
                return g1_recovery_fail(
                    error,
                    error_capacity,
                    "recovery large bound is invalid");
            }
        }
    }
    return true;
}

static bool g1_recovery_normalize_query(
    float (&normalized)[G1RecoveryFeatureCount],
    const G1RecoveryRequest& request,
    G1RecoveryStrictAuditSink* audit)
{
    const database& db = *request.db;
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        const float raw = request.raw_query[feature];
        const float offset = db.features_offset(static_cast<int>(feature));
        const float scale = db.features_scale(static_cast<int>(feature));
        if (!g1_recovery_float_is_finite(raw) ||
            !g1_recovery_float_is_finite(offset)) {
            return false;
        }
        if (g1_recovery_float_bits(scale) == UINT32_C(0x7f7fffff)) {
            normalized[feature] = 0.0f;
        } else {
            const uint32_t scale_bits = g1_recovery_float_bits(scale);
            if ((scale_bits & UINT32_C(0x80000000)) != 0U ||
                (scale_bits & UINT32_C(0x7fffffff)) == 0U ||
                !g1_recovery_float_is_finite(scale)) {
                return false;
            }
            const float difference = raw - offset;
            if (!g1_recovery_float_is_finite(difference)) return false;
            const float divided = difference / scale;
            if (!g1_recovery_float_is_finite(divided)) return false;
            normalized[feature] = divided;
        }
        if (audit != nullptr) {
            audit->normalized_query_bits[feature] =
                g1_recovery_float_bits(normalized[feature]);
        }
    }
    return true;
}

static bool g1_recovery_score_row(
    float& score,
    const database& db,
    int frame,
    const float (&normalized)[G1RecoveryFeatureCount],
    float seed)
{
    if (!g1_recovery_float_is_finite(seed) || seed < 0.0f ||
        frame < 0 || frame >= db.nframes()) {
        return false;
    }
    float accumulated = seed;
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        const float row_value =
            db.features(frame, static_cast<int>(feature));
        if (!g1_recovery_float_is_finite(row_value)) return false;
        const float difference = normalized[feature] - row_value;
        if (!g1_recovery_float_is_finite(difference)) return false;
        const float square = difference * difference;
        if (!g1_recovery_float_is_finite(square)) return false;
        const float next = accumulated + square;
        if (!g1_recovery_float_is_finite(next)) return false;
        accumulated = next;
    }
    score = accumulated;
    return true;
}

static bool g1_recovery_score_bound(
    float& score,
    const slice2d<float> minimum,
    const slice2d<float> maximum,
    int box,
    const float (&normalized)[G1RecoveryFeatureCount],
    float seed,
    float strict_limit,
    bool equality_loses)
{
    if (!g1_recovery_float_is_finite(seed) || seed < 0.0f ||
        !g1_recovery_float_is_finite(strict_limit) ||
        strict_limit < 0.0f || box < 0 || box >= minimum.rows ||
        box >= maximum.rows) {
        return false;
    }
    float accumulated = seed;
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        if ((equality_loses && accumulated >= strict_limit) ||
            (!equality_loses && accumulated > strict_limit)) {
            score = accumulated;
            return true;
        }
        const float query = normalized[feature];
        const float lower = minimum(box, static_cast<int>(feature));
        const float upper = maximum(box, static_cast<int>(feature));
        const float clamped = query < lower
            ? lower
            : (query > upper ? upper : query);
        const float difference = query - clamped;
        if (!g1_recovery_float_is_finite(difference)) return false;
        const float square = difference * difference;
        if (!g1_recovery_float_is_finite(square)) return false;
        const float next = accumulated + square;
        if (!g1_recovery_float_is_finite(next)) return false;
        accumulated = next;
    }
    score = accumulated;
    return true;
}

static bool g1_recovery_record_less(
    const G1CandidateRecord& left,
    const G1CandidateRecord& right)
{
    if (left.selected_cost < right.selected_cost) return true;
    if (right.selected_cost < left.selected_cost) return false;
    return left.selected_frame < right.selected_frame;
}

static bool g1_recovery_insert_top_six(
    G1CandidateRecord (&retained)[G1RecoveryTransitionCapacity],
    uint32_t& retained_count,
    const G1CandidateRecord& candidate)
{
    if (retained_count > G1RecoveryTransitionCapacity) return false;
    for (uint32_t index = 0U; index < retained_count; ++index) {
        if (retained[index].selected_frame == candidate.selected_frame) {
            return true;
        }
    }
    if (retained_count < G1RecoveryTransitionCapacity) {
        retained[retained_count] = candidate;
        ++retained_count;
    } else if (g1_recovery_record_less(
                   candidate, retained[retained_count - 1U])) {
        retained[retained_count - 1U] = candidate;
    } else {
        return true;
    }
    for (uint32_t index = retained_count - 1U; index > 0U; --index) {
        if (!g1_recovery_record_less(
                retained[index], retained[index - 1U])) {
            break;
        }
        const G1CandidateRecord temporary = retained[index - 1U];
        retained[index - 1U] = retained[index];
        retained[index] = temporary;
    }
    return true;
}

static bool g1_recovery_call_storage_is_valid(
    const G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    const void* audit,
    std::size_t audit_size,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0 || (error_capacity > 0 && error == nullptr)) {
        return false;
    }
    const std::size_t error_size = error_capacity > 0
        ? static_cast<std::size_t>(error_capacity)
        : 0U;
    uintptr_t error_begin = 0U;
    uintptr_t error_end = 0U;
    if (!g1_recovery_span(
            error_begin, error_end, error, error_size)) {
        return false;
    }
    if (g1_recovery_overlaps(
            &output, sizeof(output), &request, sizeof(request))) {
        return false;
    }
    if (audit_size > 0U &&
        (audit == nullptr ||
         g1_recovery_overlaps(
             &output, sizeof(output), audit, audit_size) ||
         g1_recovery_overlaps(
             &request, sizeof(request), audit, audit_size))) {
        return false;
    }
    if (error_size > 0U &&
        (g1_recovery_overlaps(
             &output, sizeof(output), error, error_size) ||
         g1_recovery_overlaps(
             &request, sizeof(request), error, error_size) ||
         (audit_size > 0U &&
          g1_recovery_overlaps(
              audit, audit_size, error, error_size)))) {
        return false;
    }
    return true;
}

static int g1_recovery_source_range(const database& db, int frame)
{
    for (int range = 0; range < db.nranges(); ++range) {
        if (frame >= db.range_starts(range) &&
            frame < db.range_stops(range)) {
            return range;
        }
    }
    return -1;
}

static int g1_recovery_executed_frame(
    const database& db,
    int selected_frame,
    int source_range)
{
    if (source_range < 0 || source_range >= db.nranges()) return -1;
    const int last = db.range_stops(source_range) - 1;
    return selected_frame < last ? selected_frame + 1 : selected_frame;
}

static bool g1_recovery_frame_is_excluded(
    const G1RecoveryRequest& request,
    int frame)
{
    if (frame == request.incumbent_frame ||
        frame == request.legacy_selected_frame) {
        return true;
    }
    const int64_t separation = frame >= request.incumbent_frame
        ? static_cast<int64_t>(frame) - request.incumbent_frame
        : static_cast<int64_t>(request.incumbent_frame) - frame;
    return separation < static_cast<int64_t>(request.ignore_surrounding);
}

static bool g1_recovery_bound_prunes(
    bool& pruned,
    const slice2d<float> minimum,
    const slice2d<float> maximum,
    int box,
    const float (&normalized)[G1RecoveryFeatureCount],
    const G1RecoveryRequest& request,
    float private_incumbent_score,
    const G1CandidateRecord (&retained)[G1RecoveryTransitionCapacity],
    uint32_t retained_count)
{
    pruned = false;
    float bound_score = 0.0f;
    if (!g1_recovery_score_bound(
            bound_score,
            minimum,
            maximum,
            box,
            normalized,
            request.transition_cost,
            private_incumbent_score,
            true)) {
        return false;
    }
    if (bound_score >= private_incumbent_score) {
        pruned = true;
        return true;
    }

    if (g1_recovery_float_bits(request.public_incumbent_cost) !=
        UINT32_C(0x7f7fffff)) {
        if (!g1_recovery_score_bound(
                bound_score,
                minimum,
                maximum,
                box,
                normalized,
                request.transition_cost,
                request.public_incumbent_cost,
                true)) {
            return false;
        }
        if (bound_score >= request.public_incumbent_cost) {
            pruned = true;
            return true;
        }
    }

    if (retained_count == G1RecoveryTransitionCapacity) {
        const float top_six_worst =
            retained[retained_count - 1U].selected_cost;
        if (!g1_recovery_score_bound(
                bound_score,
                minimum,
                maximum,
                box,
                normalized,
                request.transition_cost,
                top_six_worst,
                false)) {
            return false;
        }
        if (bound_score > top_six_worst) {
            pruned = true;
            return true;
        }
    }
    return true;
}

static bool g1_recovery_consider_frame(
    G1CandidateRecord (&retained)[G1RecoveryTransitionCapacity],
    uint32_t& retained_count,
    G1RecoveryCandidateSet& candidate_set,
    G1RecoveryStrictAuditSink& audit,
    const G1RecoveryRequest& request,
    const float (&normalized)[G1RecoveryFeatureCount],
    float private_incumbent_score,
    int frame,
    int source_range,
    bool exhaustive)
{
    if (g1_recovery_frame_is_excluded(request, frame)) return true;
    if (!g1_recovery_increment(candidate_set.work.rows_tested)) {
        return false;
    }
    if (exhaustive &&
        !g1_recovery_increment(audit.exhaustive_rows_tested)) {
        return false;
    }

    float score = 0.0f;
    if (!g1_recovery_score_row(
            score,
            *request.db,
            frame,
            normalized,
            request.transition_cost)) {
        return false;
    }
    if (!g1_recovery_increment(
            candidate_set.work.full_scores_materialized)) {
        return false;
    }
    if (!(score < private_incumbent_score)) return true;
    if (g1_recovery_float_bits(request.public_incumbent_cost) !=
            UINT32_C(0x7f7fffff) &&
        !(score < request.public_incumbent_cost)) {
        return true;
    }

    G1CandidateRecord record;
    record.kind = G1CandidateRecoveryTransition;
    record.selected_frame = frame;
    record.executed_frame = g1_recovery_executed_frame(
        *request.db, frame, source_range);
    record.source_range = source_range;
    record.selected_cost = score;
    record.recovery_rank = UINT32_MAX;
    record.transitioned = true;
    return g1_recovery_insert_top_six(
        retained, retained_count, record);
}

#if defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
static bool g1_recovery_enumerate_exhaustive(
    G1CandidateRecord (&retained)[G1RecoveryTransitionCapacity],
    uint32_t& retained_count,
    G1RecoveryCandidateSet& candidate_set,
    G1RecoveryStrictAuditSink& audit,
    const G1RecoveryRequest& request,
    const float (&normalized)[G1RecoveryFeatureCount],
    float private_incumbent_score)
{
    const database& db = *request.db;
    for (int range = 0; range < db.nranges(); ++range) {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        const int length = stop - start;
        const int eligible_length = request.ignore_range_end < length
            ? length - request.ignore_range_end
            : 0;
        const int search_stop = start + eligible_length;
        for (int frame = start; frame < search_stop; ++frame) {
            if (!g1_recovery_consider_frame(
                    retained,
                    retained_count,
                    candidate_set,
                    audit,
                    request,
                    normalized,
                    private_incumbent_score,
                    frame,
                    range,
                    true)) {
                return false;
            }
        }
    }
    return true;
}
#endif

static int g1_recovery_box_end(int frame, int box_size, int limit)
{
    const int64_t box = static_cast<int64_t>(frame / box_size);
    const int64_t next = (box + 1) * box_size;
    return next < static_cast<int64_t>(limit)
        ? static_cast<int>(next)
        : limit;
}

static bool g1_recovery_enumerate_accelerated(
    G1CandidateRecord (&retained)[G1RecoveryTransitionCapacity],
    uint32_t& retained_count,
    G1RecoveryCandidateSet& candidate_set,
    G1RecoveryStrictAuditSink& audit,
    const G1RecoveryRequest& request,
    const float (&normalized)[G1RecoveryFeatureCount],
    float private_incumbent_score)
{
    const database& db = *request.db;
    if (!g1_recovery_increment(
            candidate_set.work.accelerated_traversals)) {
        return false;
    }
    for (int range = 0; range < db.nranges(); ++range) {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        const int length = stop - start;
        const int eligible_length = request.ignore_range_end < length
            ? length - request.ignore_range_end
            : 0;
        const int search_stop = start + eligible_length;
        int frame = start;
        while (frame < search_stop) {
            const int large_box = frame / BOUND_LR_SIZE;
            const int large_end = g1_recovery_box_end(
                frame, BOUND_LR_SIZE, search_stop);
            if (!g1_recovery_increment(
                    candidate_set.work.large_bounds_tested)) {
                return false;
            }
            bool large_pruned = false;
            if (!g1_recovery_bound_prunes(
                    large_pruned,
                    db.bound_lr_min,
                    db.bound_lr_max,
                    large_box,
                    normalized,
                    request,
                    private_incumbent_score,
                    retained,
                    retained_count)) {
                return false;
            }
            if (large_pruned) {
                frame = large_end;
                continue;
            }

            while (frame < large_end) {
                const int small_box = frame / BOUND_SM_SIZE;
                const int small_end = g1_recovery_box_end(
                    frame, BOUND_SM_SIZE, large_end);
                if (!g1_recovery_increment(
                        candidate_set.work.small_bounds_tested)) {
                    return false;
                }
                bool small_pruned = false;
                if (!g1_recovery_bound_prunes(
                        small_pruned,
                        db.bound_sm_min,
                        db.bound_sm_max,
                        small_box,
                        normalized,
                        request,
                        private_incumbent_score,
                        retained,
                        retained_count)) {
                    return false;
                }
                if (small_pruned) {
                    frame = small_end;
                    continue;
                }
                while (frame < small_end) {
                    if (!g1_recovery_consider_frame(
                            retained,
                            retained_count,
                            candidate_set,
                            audit,
                            request,
                            normalized,
                            private_incumbent_score,
                            frame,
                            range,
                            false)) {
                        return false;
                    }
                    ++frame;
                }
            }
        }
    }
    return candidate_set.work.accelerated_traversals == 1U;
}

static bool g1_recovery_candidate_set_is_valid(
    const G1RecoveryCandidateSet& candidate_set,
    const G1RecoveryRequest& request,
    float private_incumbent_score,
    uint32_t retained_count,
    bool exhaustive)
{
    if (retained_count > G1RecoveryTransitionCapacity ||
        candidate_set.count > G1RecoveryTailCapacity ||
        candidate_set.count > G1CandidateAttemptCapacity ||
        candidate_set.work.full_scores_materialized !=
            candidate_set.work.rows_tested ||
        candidate_set.work.accelerated_traversals !=
            (exhaustive ? 0U : 1U) ||
        (exhaustive &&
         (candidate_set.work.large_bounds_tested != 0U ||
          candidate_set.work.small_bounds_tested != 0U))) {
        return false;
    }

    const bool append_incumbent =
        request.legacy_selected_frame != request.incumbent_frame;
    const uint32_t expected_count = retained_count +
        (append_incumbent ? 1U : 0U);
    if (candidate_set.count != expected_count) return false;

    for (uint32_t index = 0U; index < retained_count; ++index) {
        const G1CandidateRecord& record = candidate_set.records[index];
        if (record.kind != G1CandidateRecoveryTransition ||
            !record.transitioned || record.recovery_rank != index ||
            record.selected_frame < 0 ||
            record.selected_frame >= request.db->nframes() ||
            record.selected_frame == request.incumbent_frame ||
            record.selected_frame == request.legacy_selected_frame ||
            record.source_range < 0 ||
            record.source_range >= request.db->nranges() ||
            record.selected_frame <
                request.db->range_starts(record.source_range) ||
            record.selected_frame >=
                request.db->range_stops(record.source_range) ||
            record.executed_frame != g1_recovery_executed_frame(
                *request.db,
                record.selected_frame,
                record.source_range) ||
            !g1_recovery_float_is_finite(record.selected_cost) ||
            record.selected_cost < 0.0f ||
            !(record.selected_cost < private_incumbent_score) ||
            (g1_recovery_float_bits(request.public_incumbent_cost) !=
                 UINT32_C(0x7f7fffff) &&
             !(record.selected_cost < request.public_incumbent_cost))) {
            return false;
        }
        for (uint32_t previous = 0U; previous < index; ++previous) {
            if (candidate_set.records[previous].selected_frame ==
                record.selected_frame) {
                return false;
            }
        }
        if (index > 0U &&
            g1_recovery_record_less(
                record, candidate_set.records[index - 1U])) {
            return false;
        }
    }

    if (append_incumbent) {
        const G1CandidateRecord& incumbent =
            candidate_set.records[retained_count];
        const int source_range = g1_recovery_source_range(
            *request.db, request.incumbent_frame);
        if (incumbent.kind != G1CandidateIncumbent ||
            incumbent.transitioned ||
            incumbent.selected_frame != request.incumbent_frame ||
            incumbent.source_range != source_range ||
            incumbent.executed_frame != g1_recovery_executed_frame(
                *request.db, request.incumbent_frame, source_range) ||
            g1_recovery_float_bits(incumbent.selected_cost) !=
                g1_recovery_float_bits(request.public_incumbent_cost) ||
            incumbent.recovery_rank != UINT32_MAX) {
            return false;
        }
    }
    return true;
}

static bool g1_recovery_finalize_candidate_set(
    G1RecoveryCandidateSet& candidate_set,
    G1CandidateRecord (&retained)[G1RecoveryTransitionCapacity],
    uint32_t retained_count,
    const G1RecoveryRequest& request)
{
    if (retained_count > G1RecoveryTransitionCapacity) return false;
    for (uint32_t rank = 0U; rank < retained_count; ++rank) {
        retained[rank].recovery_rank = rank;
        candidate_set.records[rank] = retained[rank];
    }
    candidate_set.count = retained_count;
    if (request.legacy_selected_frame != request.incumbent_frame) {
        if (candidate_set.count >= G1RecoveryTailCapacity) return false;
        const int source_range = g1_recovery_source_range(
            *request.db, request.incumbent_frame);
        if (source_range < 0) return false;
        G1CandidateRecord incumbent;
        incumbent.kind = G1CandidateIncumbent;
        incumbent.selected_frame = request.incumbent_frame;
        incumbent.executed_frame = g1_recovery_executed_frame(
            *request.db, request.incumbent_frame, source_range);
        incumbent.source_range = source_range;
        incumbent.selected_cost = request.public_incumbent_cost;
        incumbent.recovery_rank = UINT32_MAX;
        incumbent.transitioned = false;
        candidate_set.records[candidate_set.count] = incumbent;
        ++candidate_set.count;
    }
    return true;
}

static G1RecoveryProviderStatus g1_recovery_candidates_build_internal(
    G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    G1RecoveryStrictAuditSink& audit,
    bool exhaustive,
    char* error,
    int error_capacity)
{
    if (!g1_recovery_request_is_valid(
            request, error, error_capacity)) {
        return G1RecoveryProviderGlobalError;
    }

    float normalized[G1RecoveryFeatureCount] = {};
    if (!g1_recovery_normalize_query(normalized, request, &audit)) {
        g1_recovery_fail(
            error, error_capacity, "recovery normalization failed");
        return G1RecoveryProviderGlobalError;
    }

    float private_incumbent_score = 0.0f;
    if (!g1_recovery_score_row(
            private_incumbent_score,
            *request.db,
            request.incumbent_frame,
            normalized,
            0.0f)) {
        g1_recovery_fail(
            error, error_capacity, "recovery incumbent score failed");
        return G1RecoveryProviderGlobalError;
    }
    audit.recovery_incumbent_score_bits =
        g1_recovery_float_bits(private_incumbent_score);

    G1RecoveryCandidateSet candidate_set;
    G1CandidateRecord retained[G1RecoveryTransitionCapacity] = {};
    uint32_t retained_count = 0U;
#if defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
    const bool enumerated = exhaustive
        ? g1_recovery_enumerate_exhaustive(
              retained,
              retained_count,
              candidate_set,
              audit,
              request,
              normalized,
              private_incumbent_score)
        : g1_recovery_enumerate_accelerated(
              retained,
              retained_count,
              candidate_set,
              audit,
              request,
              normalized,
              private_incumbent_score);
#else
    if (exhaustive) {
        g1_recovery_fail(
            error, error_capacity, "recovery traversal mode is invalid");
        return G1RecoveryProviderGlobalError;
    }
    const bool enumerated = g1_recovery_enumerate_accelerated(
        retained,
        retained_count,
        candidate_set,
        audit,
        request,
        normalized,
        private_incumbent_score);
#endif
    if (!enumerated) {
        g1_recovery_fail(
            error, error_capacity, "recovery traversal failed");
        return G1RecoveryProviderGlobalError;
    }
    if (!g1_recovery_finalize_candidate_set(
            candidate_set, retained, retained_count, request) ||
        !g1_recovery_candidate_set_is_valid(
            candidate_set,
            request,
            private_incumbent_score,
            retained_count,
            exhaustive)) {
        g1_recovery_fail(
            error, error_capacity, "recovery candidate set is invalid");
        return G1RecoveryProviderGlobalError;
    }
    output = candidate_set;
    return G1RecoveryProviderOk;
}

G1RecoveryProviderStatus g1_recovery_candidates_build(
    G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    char* error,
    int error_capacity)
{
    if (!g1_recovery_call_storage_is_valid(
            output,
            request,
            nullptr,
            0U,
            error,
            error_capacity)) {
        return G1RecoveryProviderGlobalError;
    }
    G1RecoveryStrictAuditSink audit;
    return g1_recovery_candidates_build_internal(
        output,
        request,
        audit,
        false,
        error,
        error_capacity);
}

#if defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
static void g1_recovery_publish_audit(
    G1RecoveryProviderAudit& output,
    const G1RecoveryStrictAuditSink& input)
{
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        output.normalized_query_bits[feature] =
            input.normalized_query_bits[feature];
    }
    output.recovery_incumbent_score_bits =
        input.recovery_incumbent_score_bits;
    output.exhaustive_rows_tested = input.exhaustive_rows_tested;
}

G1RecoveryProviderStatus g1_recovery_candidates_build_audited(
    G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    G1RecoveryProviderAudit& audit,
    char* error,
    int error_capacity)
{
    if (!g1_recovery_call_storage_is_valid(
            output,
            request,
            &audit,
            sizeof(audit),
            error,
            error_capacity)) {
        return G1RecoveryProviderGlobalError;
    }
    G1RecoveryStrictAuditSink candidate_audit;
    const G1RecoveryProviderStatus status =
        g1_recovery_candidates_build_internal(
            output,
            request,
            candidate_audit,
            false,
            error,
            error_capacity);
    if (status == G1RecoveryProviderOk) {
        g1_recovery_publish_audit(audit, candidate_audit);
    }
    return status;
}

G1RecoveryProviderStatus g1_recovery_candidates_exhaustive_for_test(
    G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    G1RecoveryProviderAudit& audit,
    char* error,
    int error_capacity)
{
    if (!g1_recovery_call_storage_is_valid(
            output,
            request,
            &audit,
            sizeof(audit),
            error,
            error_capacity)) {
        return G1RecoveryProviderGlobalError;
    }
    G1RecoveryStrictAuditSink candidate_audit;
    const G1RecoveryProviderStatus status =
        g1_recovery_candidates_build_internal(
            output,
            request,
            candidate_audit,
            true,
            error,
            error_capacity);
    if (status == G1RecoveryProviderOk) {
        g1_recovery_publish_audit(audit, candidate_audit);
    }
    return status;
}
#endif
