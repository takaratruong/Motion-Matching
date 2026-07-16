#include "g1_candidate_recovery.h"

int main()
{
    G1RecoveryCandidateSet output;
    G1RecoveryRequest request;
    G1RecoveryProviderAudit audit;
    char error[128] = {};
    return g1_recovery_candidates_exhaustive_for_test(
               output, request, audit, error, 128) ==
                   G1RecoveryProviderGlobalError
               ? 0
               : 1;
}
