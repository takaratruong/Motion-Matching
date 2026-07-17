#pragma once

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM) && defined(G1_CANDIDATE_RECOVERY_ENABLE_TEST_SEAM)

#include "g1_frame_transaction.h"

#include <cstdint>
#include <cstdio>

struct G1CandidateTraceFile
{
    std::FILE* stream = nullptr;
    bool header_written = false;
};

bool g1_candidate_trace_open(
    G1CandidateTraceFile& file,
    const char* path,
    char* error,
    int error_capacity);

bool g1_candidate_trace_append_after_transaction(
    G1CandidateTraceFile& file,
    uint32_t presentation_frame,
    const G1CandidateCertificationTrace& trace,
    char* error,
    int error_capacity);

void g1_candidate_trace_close(G1CandidateTraceFile& file);

#endif
