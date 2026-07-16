#pragma once

#include "database.h"

#include <cfloat>
#include <cstdint>

static constexpr uint32_t G1RecoveryFeatureCount = 31U;
static constexpr uint32_t G1CandidateAttemptCapacity = 8U;
static constexpr uint32_t G1RecoveryTransitionCapacity = 6U;
static constexpr uint32_t G1RecoveryTailCapacity = 7U;

enum G1CandidateKind : uint32_t
{
    G1CandidateLegacy = 0U,
    G1CandidateRecoveryTransition,
    G1CandidateIncumbent,
};

struct G1CandidateRecord
{
    G1CandidateKind kind = G1CandidateIncumbent;
    int selected_frame = -1;
    int executed_frame = -1;
    int source_range = -1;
    float selected_cost = FLT_MAX;
    uint32_t recovery_rank = UINT32_MAX;
    bool transitioned = false;
};

struct G1RecoveryRequest
{
    const database* db = nullptr;
    float raw_query[G1RecoveryFeatureCount] = {};
    int incumbent_frame = -1;
    int legacy_selected_frame = -1;
    float transition_cost = 0.0f;
    float public_incumbent_cost = FLT_MAX;
    int ignore_range_end = 20;
    int ignore_surrounding = 20;
};

struct G1RecoveryWork
{
    uint32_t accelerated_traversals = 0U;
    uint32_t large_bounds_tested = 0U;
    uint32_t small_bounds_tested = 0U;
    uint32_t rows_tested = 0U;
    uint32_t full_scores_materialized = 0U;
};

struct G1RecoveryCandidateSet
{
    G1CandidateRecord records[G1CandidateAttemptCapacity] = {};
    uint32_t count = 0U;
    G1RecoveryWork work;
};

enum G1RecoveryProviderStatus : uint32_t
{
    G1RecoveryProviderOk = 0U,
    G1RecoveryProviderGlobalError,
};

using G1RecoveryProvider = G1RecoveryProviderStatus (*)(
    G1RecoveryCandidateSet&,
    const G1RecoveryRequest&,
    char*,
    int);

G1RecoveryProviderStatus g1_recovery_candidates_build(
    G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    char* error,
    int error_capacity);

#if defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)
struct G1RecoveryProviderAudit
{
    uint32_t normalized_query_bits[G1RecoveryFeatureCount] = {};
    uint32_t recovery_incumbent_score_bits = 0U;
    uint32_t exhaustive_rows_tested = 0U;
};

G1RecoveryProviderStatus g1_recovery_candidates_build_audited(
    G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    G1RecoveryProviderAudit& audit,
    char* error,
    int error_capacity);

G1RecoveryProviderStatus g1_recovery_candidates_exhaustive_for_test(
    G1RecoveryCandidateSet& output,
    const G1RecoveryRequest& request,
    G1RecoveryProviderAudit& audit,
    char* error,
    int error_capacity);
#endif
