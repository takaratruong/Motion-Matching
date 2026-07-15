#ifndef NDEBUG
#error "Carry fast-math validation requires NDEBUG"
#endif

#ifndef __FAST_MATH__
#error "Carry fast-math validation requires -ffast-math"
#endif

#include "interaction_carry.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace {

template<class Function>
bool throws_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
    }
    return false;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

interaction::Transform hand_world(
    const interaction::Pose& pose,
    interaction::Hand hand) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    const size_t bone = hand == interaction::Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
    return {world.positions[bone], world.rotations[bone]};
}

interaction::Transform object_world_from_hold_pose(
    const interaction::Pose& hold,
    interaction::Hand hand,
    const interaction::GraspAffordance& affordance) {
    return interaction::compose(
        hand_world(hold, hand),
        interaction::inverse(affordance.hand_in_object));
}

void test_nonfinite_inputs_are_rejected() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    require(affordance != nullptr, "fixture affordance is missing");
    const Pose hold = pose_at_frame(
        fixture.database, fixture.database.range_stops.at(1) - 1);
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, *affordance);
    const CarryRanges ranges = classify_carry_ranges(fixture.database);

    CarryController invalid_start(
        fixture.database, fixture.features, ranges);
    Pose nonfinite_hold = hold;
    nonfinite_hold.positions[g1_skeleton::Simulation].x =
        std::numeric_limits<float>::quiet_NaN();
    require(
        throws_invalid_argument([&] {
            invalid_start.start(
                nonfinite_hold, Hand::Right, *affordance, object);
        }),
        "Carry accepted a nonfinite final Hold pose under fast-math");

    CarryController invalid_update(
        fixture.database, fixture.features, ranges);
    invalid_update.start(hold, Hand::Right, *affordance, object);
    require(
        throws_invalid_argument([&] {
            (void)invalid_update.update(
                fixture.locomotion,
                std::numeric_limits<float>::quiet_NaN());
        }),
        "Carry accepted a nonfinite dt under fast-math");
}

}  // namespace

int main() {
    try {
        test_nonfinite_inputs_are_rejected();
    } catch (...) {
        return 1;
    }
    return 0;
}
