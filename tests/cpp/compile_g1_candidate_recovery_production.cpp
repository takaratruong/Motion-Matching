#include "g1_candidate_recovery.h"

#include <type_traits>

static_assert(
    std::is_same<
        G1RecoveryProvider,
        decltype(&g1_recovery_candidates_build)>::value,
    "production recovery provider signature changed");

int main()
{
    G1RecoveryCandidateSet output;
    G1RecoveryRequest request;
    char error[128] = {};
    const G1RecoveryProvider provider = &g1_recovery_candidates_build;
    return provider(output, request, error, 128) ==
                   G1RecoveryProviderGlobalError
               ? 0
               : 1;
}
