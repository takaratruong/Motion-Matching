#include "g1_ik_runtime.h"

#include <type_traits>

using G1IkSafeStopSignature = bool (*)(
    G1IkSafeStopHandoff&, bool, vec3, char*, int);
using G1RootReachPlannerSignature = bool (*)(
    G1RootReachPlan&,
    const slice1d<vec3>,
    const slice1d<quat>,
    const slice1d<int>,
    const slice1d<bool>,
    const G1FootTarget&,
    const G1FootTarget&,
    char*,
    int);
using G1RootReachApplySignature = bool (*)(
    float&, float, const G1RootReachPlan&);

static_assert(
    std::is_same<decltype(&g1_ik_safe_stop_handoff),
                 G1IkSafeStopSignature>::value,
    "production runtime exposes the checked safe-stop handoff");
static_assert(
    std::is_same<decltype(&g1_plan_recorded_contact_root_reach),
                 G1RootReachPlannerSignature>::value,
    "production runtime exposes the public root-reach planner");
static_assert(
    std::is_same<decltype(&g1_apply_root_reach_plan_y),
                 G1RootReachApplySignature>::value,
    "production runtime exposes the strict root-Y apply helper");
static_assert(
    std::is_same<decltype(&G1IkFrameResult::root_reach),
                 G1RootReachPlan G1IkFrameResult::*>::value,
    "production frame results publish the value-only root-reach plan");

int main()
{
    return G1SwingLiftCandidateCount == 41 ? 0 : 1;
}
