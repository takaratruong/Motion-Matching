#include "g1_ik_runtime.h"

#include <type_traits>

using G1IkSafeStopSignature = bool (*)(
    G1IkSafeStopHandoff&, bool, vec3, char*, int);

static_assert(
    std::is_same<decltype(&g1_ik_safe_stop_handoff),
                 G1IkSafeStopSignature>::value,
    "production runtime exposes the checked safe-stop handoff");

int main()
{
    return G1SwingLiftCandidateCount == 41 ? 0 : 1;
}
