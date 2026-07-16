#pragma once

#include "g1_frame_transaction.h"

G1FrameStageOutcome g1_controller_frame_stage_run(
    G1FrameTransactionStage stage,
    g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity);
