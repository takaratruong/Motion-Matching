#include "g1_frame_transaction.h"

#if (defined(G1_FRAME_NEGATIVE_VOID_CONTEXT) + \
     defined(G1_FRAME_NEGATIVE_PUBLICATION_CONTEXT) + \
     defined(G1_FRAME_NEGATIVE_ACCEPTED_CONTEXT)) != 1
#error "select exactly one forbidden G1 frame runner context"
#endif

#if defined(G1_FRAME_NEGATIVE_VOID_CONTEXT)
static G1FrameStageOutcome runner_with_forbidden_context(
    G1FrameTransactionStage,
    g1_controller_state&,
    G1FrameTransactionScratch&,
    void*,
    char*, int)
#elif defined(G1_FRAME_NEGATIVE_PUBLICATION_CONTEXT)
static G1FrameStageOutcome runner_with_forbidden_context(
    G1FrameTransactionStage,
    g1_controller_state&,
    G1FrameTransactionScratch&,
    G1FramePublication*,
    char*, int)
#else
static G1FrameStageOutcome runner_with_forbidden_context(
    G1FrameTransactionStage,
    g1_controller_state&,
    G1FrameTransactionScratch&,
    g1_controller_state*,
    char*, int)
#endif
{
    return G1FrameStageContinue;
}

G1FrameStageRunner forbidden_runner = &runner_with_forbidden_context;
