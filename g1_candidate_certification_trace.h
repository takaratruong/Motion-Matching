#pragma once

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM) && defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)

#include "g1_frame_transaction.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

struct G1CandidateTraceFile
{
    std::FILE* stream = nullptr;
    bool header_written = false;
};

static inline bool g1_candidate_trace_open(
    G1CandidateTraceFile& file,
    const char* path,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0 ||
        (error == nullptr && error_capacity > 0)) {
        return false;
    }
    const std::size_t error_bytes =
        error != nullptr && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U;
    if (g1_ik_memory_ranges_overlap(
            error, error_bytes, &file, sizeof(file))) {
        return false;
    }
    if (path != nullptr) {
        if (g1_ik_memory_ranges_overlap(
                path, 1U, &file, sizeof(file))) {
            return false;
        }
        const std::size_t path_length = std::strlen(path);
        if (path_length == std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        const std::size_t path_bytes = path_length + 1U;
        if (g1_ik_memory_ranges_overlap(
                error, error_bytes, path, path_bytes) ||
            g1_ik_memory_ranges_overlap(
                path, path_bytes, &file, sizeof(file))) {
            return false;
        }
    }
    const auto fail = [error, error_capacity](const char* message) {
        if (error != nullptr && error_capacity > 0) {
            std::snprintf(
                error,
                static_cast<std::size_t>(error_capacity),
                "%s",
                message);
        }
        return false;
    };
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    if (file.stream != nullptr || file.header_written) {
        return fail("candidate trace file is already open");
    }
    if (path == nullptr || path[0] != '/') {
        return fail("candidate trace path must be nonempty and absolute");
    }
    for (const unsigned char* cursor =
             reinterpret_cast<const unsigned char*>(path);
         *cursor != 0U;
         ++cursor) {
        if (*cursor == '\r' || *cursor == '\n') {
            return fail("candidate trace path contains a line break");
        }
    }
    std::FILE* const stream = std::fopen(path, "wb");
    if (stream == nullptr) {
        return fail("candidate trace file could not be opened");
    }
    file.stream = stream;
    file.header_written = false;
    return true;
}

static inline bool g1_candidate_trace_append_after_transaction(
    G1CandidateTraceFile& file,
    uint32_t presentation_frame,
    const G1CandidateCertificationTrace& trace,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0 ||
        (error == nullptr && error_capacity > 0)) {
        return false;
    }
    const std::size_t error_bytes =
        error != nullptr && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U;
    if (g1_ik_memory_ranges_overlap(
            error, error_bytes, &file, sizeof(file)) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes, &trace, sizeof(trace)) ||
        g1_ik_memory_ranges_overlap(
            &file, sizeof(file), &trace, sizeof(trace))) {
        return false;
    }
    const auto diagnostic_overlaps_database = [error, error_bytes](
        const database& db) {
        const g1_controller_state_memory_range diagnostic = {
            error, error_bytes
        };
        if (g1_controller_state_ranges_overlap(
                diagnostic, {&db, sizeof(db)})) {
            return true;
        }
#define G1_CANDIDATE_TRACE_DATABASE_OWNER(owner) \
        if (g1_controller_state_source_owner_overlaps( \
                diagnostic, owner)) return true
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.bone_positions);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.bone_velocities);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.bone_rotations);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.bone_angular_velocities);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.bone_parents);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.range_starts);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.range_stops);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.features);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.features_offset);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.features_scale);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.terrain_features);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.contact_states);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.bound_sm_min);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.bound_sm_max);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.bound_lr_min);
        G1_CANDIDATE_TRACE_DATABASE_OWNER(db.bound_lr_max);
#undef G1_CANDIDATE_TRACE_DATABASE_OWNER
        return false;
    };
    if (trace.recovery_request.db != nullptr &&
        diagnostic_overlaps_database(*trace.recovery_request.db)) {
        return false;
    }
    const auto fail = [error, error_capacity](const char* message) {
        if (error != nullptr && error_capacity > 0) {
            std::snprintf(
                error,
                static_cast<std::size_t>(error_capacity),
                "%s",
                message);
        }
        return false;
    };
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    if (file.stream == nullptr) {
        return fail("candidate trace file is not open");
    }
    if (trace.attempt_count == 0U ||
        trace.attempt_count > G1CandidateAttemptCapacity ||
        trace.recovery_set.count > G1RecoveryTailCapacity ||
        trace.legacy_traversals > 1U ||
        trace.recovery_provider_calls > 1U) {
        return fail("candidate trace counters exceed bounded ownership");
    }
    const auto request_is_reset = [](
        const G1RecoveryRequest& request) {
        if (request.db != nullptr ||
            request.incumbent_frame != -1 ||
            request.legacy_selected_frame != -1 ||
            terrain_float_bits(request.transition_cost) != 0U ||
            terrain_float_bits(request.public_incumbent_cost) !=
                terrain_float_bits(FLT_MAX) ||
            request.ignore_range_end != 20 ||
            request.ignore_surrounding != 20) {
            return false;
        }
        for (uint32_t feature = 0U;
             feature < G1RecoveryFeatureCount;
             ++feature) {
            if (terrain_float_bits(request.raw_query[feature]) != 0U) {
                return false;
            }
        }
        return true;
    };
    const auto recovery_work_is_reset = [](
        const G1RecoveryWork& work) {
        return work.accelerated_traversals == 0U &&
               work.large_bounds_tested == 0U &&
               work.small_bounds_tested == 0U &&
               work.rows_tested == 0U &&
               work.full_scores_materialized == 0U;
    };
    if (!trace.recovery_request_available) {
        if (trace.recovery_provider_calls != 0U ||
            trace.recovery_set.count != 0U ||
            trace.attempt_count != 1U ||
            !request_is_reset(trace.recovery_request) ||
            !recovery_work_is_reset(trace.recovery_set.work)) {
            return fail("candidate trace has noncanonical recovery evidence without a request");
        }
    } else {
        if (trace.recovery_provider_calls != 1U ||
            trace.recovery_request.db == nullptr ||
            trace.attempt_count > trace.recovery_set.count + 1U ||
            !g1_frame_recovery_set_is_valid(
                trace.recovery_set, trace.recovery_request)) {
            return fail("candidate trace request or accelerated set is not authentic");
        }
    }
    if (trace.attempt_count - 1U > trace.recovery_set.count) {
        return fail("candidate trace is missing an attempted tail prefix");
    }

    const auto records_equal = [](
        const G1CandidateRecord& first,
        const G1CandidateRecord& second) {
        return first.kind == second.kind &&
               first.selected_frame == second.selected_frame &&
               first.executed_frame == second.executed_frame &&
               first.source_range == second.source_range &&
               terrain_float_bits(first.selected_cost) ==
                   terrain_float_bits(second.selected_cost) &&
               first.recovery_rank == second.recovery_rank &&
               first.transitioned == second.transitioned;
    };
    const auto same_identity = [](
        const G1CandidateRecord& first,
        const G1CandidateRecord& second) {
        return first.kind == second.kind &&
               first.selected_frame == second.selected_frame &&
               first.executed_frame == second.executed_frame &&
               first.source_range == second.source_range &&
               first.recovery_rank == second.recovery_rank &&
               first.transitioned == second.transitioned;
    };
    const auto owner_matches_kind = [](
        G1CandidateKind kind,
        G1CandidateScoreOwner owner) {
        return (kind == G1CandidateLegacy &&
                owner == G1CandidateScoreLegacy) ||
               (kind == G1CandidateRecoveryTransition &&
                owner == G1CandidateScoreStrictRecovery) ||
               (kind == G1CandidateIncumbent &&
                owner == G1CandidateScoreIncumbent);
    };
    const auto enum_word = [](const auto& value) {
        using Enum = std::remove_cv_t<
            std::remove_reference_t<decltype(value)> >;
        using Word = std::underlying_type_t<Enum>;
        Word word = 0;
        std::memcpy(&word, &value, sizeof(word));
        return word;
    };
    using RejectionStageWord =
        std::underlying_type_t<G1FrameRejectionStage>;
    using StopReasonWord = std::underlying_type_t<G1IkStopReason>;
    const auto candidate_record_is_reset = [](
        const G1CandidateRecord& record) {
        return record.kind == G1CandidateIncumbent &&
               record.selected_frame == -1 &&
               record.executed_frame == -1 &&
               record.source_range == -1 &&
               terrain_float_bits(record.selected_cost) ==
                   terrain_float_bits(FLT_MAX) &&
               record.recovery_rank == UINT32_MAX &&
               !record.transitioned;
    };
    const auto attempt_record_is_reset = [
        &candidate_record_is_reset,
        &enum_word](
        const G1CandidateAttemptTraceRecord& record) {
        const auto stage_word = enum_word(record.rejection_stage);
        const auto reason_word = enum_word(record.stop_reason);
        return candidate_record_is_reset(record.candidate) &&
               record.score_owner == G1CandidateScoreUnassigned &&
               record.common == G1CandidateDispositionNotRun &&
               record.raw == G1CandidateDispositionNotRun &&
               record.ik == G1CandidateDispositionNotRun &&
               stage_word == static_cast<RejectionStageWord>(
                   G1FrameRejectNone) &&
               reason_word == static_cast<StopReasonWord>(
                   G1IkStopNone);
    };
    for (uint32_t index = trace.attempt_count;
         index < G1CandidateAttemptCapacity;
         ++index) {
        if (!attempt_record_is_reset(trace.attempts[index])) {
            return fail("candidate trace has hidden evidence after its attempted prefix");
        }
    }
    for (uint32_t index = trace.recovery_set.count;
         index < G1CandidateAttemptCapacity;
         ++index) {
        if (!candidate_record_is_reset(trace.recovery_set.records[index])) {
            return fail("candidate trace has hidden evidence after its recovery set");
        }
    }

    const G1CandidateAttemptTraceRecord& slot_zero = trace.attempts[0];
    if (!((trace.legacy_traversals == 1U &&
           slot_zero.candidate.kind == G1CandidateLegacy &&
           slot_zero.score_owner == G1CandidateScoreLegacy) ||
          (trace.legacy_traversals == 0U &&
           slot_zero.candidate.kind == G1CandidateIncumbent &&
           slot_zero.score_owner == G1CandidateScoreIncumbent))) {
        return fail("candidate trace slot zero disagrees with traversal and score ownership");
    }
    for (uint32_t index = 1U; index < trace.attempt_count; ++index) {
        const G1CandidateAttemptTraceRecord& attempt = trace.attempts[index];
        const G1CandidateRecord& tail = trace.recovery_set.records[index - 1U];
        if (!records_equal(attempt.candidate, tail) ||
            !owner_matches_kind(attempt.candidate.kind, attempt.score_owner)) {
            return fail("candidate trace attempted tail is not an exact prefix");
        }
    }
    for (uint32_t index = 0U; index < trace.recovery_set.count; ++index) {
        const G1CandidateRecord& record = trace.recovery_set.records[index];
        if (same_identity(slot_zero.candidate, record)) {
            return fail("candidate trace contains a duplicate slot-zero candidate");
        }
        for (uint32_t previous = 0U; previous < index; ++previous) {
            if (same_identity(
                    trace.recovery_set.records[previous], record)) {
                return fail("candidate trace contains duplicate tail candidates");
            }
        }
    }

    const auto rejection_pair_is_valid = [](
        G1FrameRejectionStage stage,
        G1IkStopReason reason) {
        switch (stage) {
        case G1FrameRejectFootprint:
            return reason == G1IkStopFootprintBlocked ||
                   reason == G1IkStopFootprintOutsideDomain ||
                   reason == G1IkStopFootprintBudgetExceeded;
        case G1FrameRejectLandingPatch:
            return reason == G1IkStopLandingPatchUnavailable;
        case G1FrameRejectIkCandidate:
            return reason == G1IkStopTargetUnreachable ||
                   reason == G1IkStopNoSwingCandidate;
        case G1FrameRejectPoseCertificate:
            return reason == G1IkStopPoseClearanceRejected;
        default:
            return false;
        }
    };
    uint32_t expected_raw_evaluations = 0U;
    uint32_t expected_ik_evaluations = 0U;
    bool dual_acceptance_seen = false;
    bool slot_zero_finite = false;
    for (uint32_t index = 0U; index < trace.attempt_count; ++index) {
        const G1CandidateAttemptTraceRecord& attempt = trace.attempts[index];
        const auto stage_word = enum_word(attempt.rejection_stage);
        const auto reason_word = enum_word(attempt.stop_reason);
        if (stage_word < static_cast<RejectionStageWord>(
                G1FrameRejectNone) ||
            stage_word > static_cast<RejectionStageWord>(
                G1FrameRejectPoseCertificate) ||
            reason_word < static_cast<StopReasonWord>(
                G1IkStopNone) ||
            reason_word > static_cast<StopReasonWord>(
                G1IkStopPoseClearanceRejected)) {
            return fail("candidate trace rejection enum word is outside its exact range");
        }
        const G1FrameRejectionStage rejection_stage =
            static_cast<G1FrameRejectionStage>(stage_word);
        const G1IkStopReason stop_reason =
            static_cast<G1IkStopReason>(reason_word);
        const bool finite =
            (attempt.common == G1CandidateDispositionFiniteRejected &&
             attempt.raw == G1CandidateDispositionNotRun &&
             attempt.ik == G1CandidateDispositionNotRun) ||
            (attempt.common == G1CandidateDispositionAccepted &&
             attempt.raw == G1CandidateDispositionFiniteRejected &&
             attempt.ik == G1CandidateDispositionNotRun) ||
            (attempt.common == G1CandidateDispositionAccepted &&
             attempt.raw == G1CandidateDispositionAccepted &&
             attempt.ik == G1CandidateDispositionFiniteRejected);
        const bool dual_accepted =
            attempt.common == G1CandidateDispositionAccepted &&
            attempt.raw == G1CandidateDispositionAccepted &&
            attempt.ik == G1CandidateDispositionAccepted;
        if (!finite && !dual_accepted) {
            return fail("candidate trace has an impossible disposition grammar");
        }
        if (attempt.common == G1CandidateDispositionAccepted) {
            ++expected_raw_evaluations;
            if (attempt.raw == G1CandidateDispositionAccepted) {
                ++expected_ik_evaluations;
            }
        }
        if (finite) {
            if (!rejection_pair_is_valid(
                    rejection_stage, stop_reason)) {
                return fail("candidate trace finite rejection evidence is not canonical");
            }
        } else {
            if (rejection_stage != G1FrameRejectNone ||
                stop_reason != G1IkStopNone ||
                dual_acceptance_seen ||
                index + 1U != trace.attempt_count) {
                return fail("candidate trace dual acceptance is not unique and terminal");
            }
            dual_acceptance_seen = true;
        }
        if (index == 0U) slot_zero_finite = finite;
    }
    if (trace.common_evaluations != trace.attempt_count ||
        trace.raw_evaluations != expected_raw_evaluations ||
        trace.ik_evaluations != expected_ik_evaluations) {
        return fail("candidate trace evaluation counters disagree with dispositions");
    }
    const bool scheduled_recovery =
        trace.legacy_traversals == 1U &&
        slot_zero.candidate.kind == G1CandidateLegacy &&
        slot_zero.score_owner == G1CandidateScoreLegacy &&
        trace.recovery_request.legacy_selected_frame ==
            slot_zero.candidate.selected_frame;
    const bool unscheduled_recovery =
        trace.legacy_traversals == 0U &&
        slot_zero.candidate.kind == G1CandidateIncumbent &&
        slot_zero.score_owner == G1CandidateScoreIncumbent &&
        !slot_zero.candidate.transitioned &&
        trace.recovery_request.incumbent_frame ==
            slot_zero.candidate.selected_frame &&
        trace.recovery_request.legacy_selected_frame ==
            slot_zero.candidate.selected_frame &&
        trace.recovery_request.ignore_range_end == 20 &&
        trace.recovery_request.ignore_surrounding == 20 &&
        g1_frame_float_bits_equal(
            trace.recovery_request.public_incumbent_cost,
            slot_zero.candidate.selected_cost);
    if (trace.recovery_request_available &&
        (!slot_zero_finite ||
         !g1_frame_candidate_record_is_valid(
             slot_zero.candidate,
             *trace.recovery_request.db) ||
         scheduled_recovery == unscheduled_recovery)) {
        return fail("candidate trace recovery is not bound to its finite legacy slot zero");
    }
    if (!trace.recovery_request_available && slot_zero_finite &&
        slot_zero.candidate.kind == G1CandidateLegacy) {
        return fail("candidate trace finite legacy slot zero lacks its recovery request");
    }
    if (!dual_acceptance_seen &&
        trace.attempt_count != trace.recovery_set.count + 1U) {
        return fail("candidate trace finite completion does not exhaust its tail");
    }

    G1RecoveryCandidateSet exhaustive;
    uint32_t exhaustive_count = 0U;
    if (trace.recovery_request_available) {
        G1RecoveryProviderAudit audit;
        const G1RecoveryProviderStatus oracle_status =
            g1_recovery_candidates_exhaustive_for_test(
                exhaustive,
                trace.recovery_request,
                audit,
                error,
                error_capacity);
        if (oracle_status != G1RecoveryProviderOk) return false;
        exhaustive_count = exhaustive.count;
        if (exhaustive.count != trace.recovery_set.count) {
            return fail("candidate trace accelerated/exhaustive count mismatch");
        }
        for (uint32_t index = 0U; index < exhaustive.count; ++index) {
            if (!records_equal(
                    trace.recovery_set.records[index],
                    exhaustive.records[index])) {
                return fail("candidate trace accelerated/exhaustive record mismatch");
            }
        }
    }

    const auto candidate_kind_text = [](G1CandidateKind kind) {
        switch (kind) {
        case G1CandidateLegacy: return "legacy";
        case G1CandidateRecoveryTransition: return "strict-recovery";
        case G1CandidateIncumbent: return "incumbent";
        default: return static_cast<const char*>(nullptr);
        }
    };
    const auto score_owner_text = [](G1CandidateScoreOwner owner) {
        switch (owner) {
        case G1CandidateScoreLegacy: return "legacy";
        case G1CandidateScoreStrictRecovery: return "strict-recovery";
        case G1CandidateScoreIncumbent: return "incumbent";
        default: return static_cast<const char*>(nullptr);
        }
    };
    const auto disposition_text = [](G1CandidateDisposition disposition) {
        switch (disposition) {
        case G1CandidateDispositionNotRun: return "not-run";
        case G1CandidateDispositionAccepted: return "accepted";
        case G1CandidateDispositionFiniteRejected:
            return "finite-rejected";
        case G1CandidateDispositionGlobalError: return "global-error";
        default: return static_cast<const char*>(nullptr);
        }
    };
    const auto stop_reason_text = [](G1IkStopReason reason) {
        if (reason < G1IkStopNone ||
            reason > G1IkStopPoseClearanceRejected) {
            return static_cast<const char*>(nullptr);
        }
        return g1_ik_stop_reason_name(reason);
    };

    char block[(G1CandidateAttemptCapacity + 2U) * 512U] = {};
    std::size_t used = 0U;
    const auto append = [&block, &used](const char* format, auto... values) {
        if (used >= sizeof(block)) return false;
        const int written = std::snprintf(
            block + used,
            sizeof(block) - used,
            format,
            values...);
        if (written < 0 ||
            static_cast<std::size_t>(written) >= sizeof(block) - used) {
            return false;
        }
        used += static_cast<std::size_t>(written);
        return true;
    };
    if (!file.header_written &&
        !append(
            "%s\n",
            "presentation_frame\tcandidate_slot\tcandidate_kind\t"
            "score_owner\tselected_frame\texecuted_frame\t"
            "source_range\tcost_bits_hex\trecovery_rank\tattempted\t"
            "common\traw\tik\trejection_stage\tstop_reason\t"
            "legacy_traversals\trecovery_provider_calls\t"
            "recovery_traversals\taccelerated_count\t"
            "exhaustive_count\toracle_equal")) {
        return fail("candidate trace header exceeds its fixed buffer");
    }

    const uint32_t row_count = trace.recovery_set.count + 1U;
    for (uint32_t slot = 0U; slot < row_count; ++slot) {
        const G1CandidateRecord& candidate = slot == 0U
            ? slot_zero.candidate
            : trace.recovery_set.records[slot - 1U];
        const bool attempted = slot == 0U || slot < trace.attempt_count;
        const G1CandidateAttemptTraceRecord* const attempt = attempted
            ? &trace.attempts[slot]
            : nullptr;
        const G1CandidateScoreOwner owner = slot == 0U
            ? slot_zero.score_owner
            : (candidate.kind == G1CandidateRecoveryTransition
                   ? G1CandidateScoreStrictRecovery
                   : (candidate.kind == G1CandidateIncumbent
                          ? G1CandidateScoreIncumbent
                          : G1CandidateScoreUnassigned));
        const char* const kind_text = candidate_kind_text(candidate.kind);
        const char* const owner_text = score_owner_text(owner);
        const char* const common_text = attempted
            ? disposition_text(attempt->common)
            : "not-run";
        const char* const raw_text = attempted
            ? disposition_text(attempt->raw)
            : "not-run";
        const char* const ik_text = attempted
            ? disposition_text(attempt->ik)
            : "not-run";
        G1FrameRejectionStage rejection_stage = G1FrameRejectNone;
        G1IkStopReason stop_reason = G1IkStopNone;
        if (attempted) {
            const auto stage_word = enum_word(attempt->rejection_stage);
            const auto reason_word = enum_word(attempt->stop_reason);
            if (stage_word < static_cast<RejectionStageWord>(
                    G1FrameRejectNone) ||
                stage_word > static_cast<RejectionStageWord>(
                    G1FrameRejectPoseCertificate) ||
                reason_word < static_cast<StopReasonWord>(
                    G1IkStopNone) ||
                reason_word > static_cast<StopReasonWord>(
                    G1IkStopPoseClearanceRejected)) {
                return fail("candidate trace rejection enum word is outside its exact range");
            }
            rejection_stage =
                static_cast<G1FrameRejectionStage>(stage_word);
            stop_reason = static_cast<G1IkStopReason>(reason_word);
        }
        const char* const rejection_text = attempted
            ? g1_frame_rejection_stage_name(rejection_stage)
            : "none";
        const char* const stop_text = attempted
            ? stop_reason_text(stop_reason)
            : "none";
        if (kind_text == nullptr || owner_text == nullptr ||
            common_text == nullptr || raw_text == nullptr ||
            ik_text == nullptr ||
            stop_text == nullptr ||
            (attempted && !owner_matches_kind(candidate.kind, owner))) {
            return fail("candidate trace row contains an invalid enum owner");
        }
        if (!append(
                "%u\t%u\t%s\t%s\t%d\t%d\t%d\t%08x\t%u\t%u\t"
                "%s\t%s\t%s\t%s\t%s\t%u\t%u\t%u\t%u\t%u\t1\n",
                static_cast<unsigned>(presentation_frame),
                static_cast<unsigned>(slot),
                kind_text,
                owner_text,
                candidate.selected_frame,
                candidate.executed_frame,
                candidate.source_range,
                static_cast<unsigned>(
                    terrain_float_bits(candidate.selected_cost)),
                static_cast<unsigned>(candidate.recovery_rank),
                attempted ? 1U : 0U,
                common_text,
                raw_text,
                ik_text,
                rejection_text,
                stop_text,
                static_cast<unsigned>(trace.legacy_traversals),
                static_cast<unsigned>(trace.recovery_provider_calls),
                static_cast<unsigned>(
                    trace.recovery_set.work.accelerated_traversals),
                static_cast<unsigned>(trace.recovery_set.count),
                static_cast<unsigned>(exhaustive_count))) {
            return fail("candidate trace transaction exceeds its fixed buffer");
        }
    }
    if (std::fwrite(block, 1U, used, file.stream) != used) {
        return fail("candidate trace transaction write failed");
    }
    if (std::fflush(file.stream) != 0) {
        return fail("candidate trace transaction flush failed");
    }
    file.header_written = true;
    return true;
}

static inline void g1_candidate_trace_close(
    G1CandidateTraceFile& file)
{
    if (file.stream != nullptr) {
        std::fclose(file.stream);
        file.stream = nullptr;
    }
    file.header_written = false;
}

#endif
