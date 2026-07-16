#include "sonic/cpp/g1_runtime.h"

int g1_runtime_linkage_peer_mode();

int main()
{
    g1_runtime_step_request request;
    return request.mode == G1RuntimeDirect &&
                   g1_runtime_linkage_peer_mode() == G1RuntimeDirect
        ? 0
        : 1;
}
