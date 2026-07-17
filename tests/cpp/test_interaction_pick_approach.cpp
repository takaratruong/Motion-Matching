#include "interaction_pick_approach.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

constexpr float kTolerance = 2.0e-5F;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(
    float actual,
    float expected,
    float tolerance,
    const char* message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void require_near(
    vec3 actual,
    vec3 expected,
    float tolerance,
    const char* message) {
    require_near(actual.x, expected.x, tolerance, message);
    require_near(actual.y, expected.y, tolerance, message);
    require_near(actual.z, expected.z, tolerance, message);
}

template<class Function>
void require_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(message);
}

float planar_distance(vec3 left, vec3 right) {
    const float x = left.x - right.x;
    const float z = left.z - right.z;
    return std::sqrt(x * x + z * z);
}

float quaternion_sign_distance(quat left, quat right) {
    const float same = std::sqrt(
        (left.w - right.w) * (left.w - right.w) +
        (left.x - right.x) * (left.x - right.x) +
        (left.y - right.y) * (left.y - right.y) +
        (left.z - right.z) * (left.z - right.z));
    const float opposite = std::sqrt(
        (left.w + right.w) * (left.w + right.w) +
        (left.x + right.x) * (left.x + right.x) +
        (left.y + right.y) * (left.y + right.y) +
        (left.z + right.z) * (left.z + right.z));
    return std::min(same, opposite);
}

interaction::InteractionTarget make_target(
    interaction::Hand hand = interaction::Hand::Right,
    vec3 object_position = vec3(0.0F, 0.75F, 2.0F),
    quat object_rotation = quat()) {
    interaction::InteractionTarget target{};
    target.handle = {4U, 2U};
    target.object_world = {object_position, object_rotation};
    target.object_dimensions = vec3(0.20F, 0.30F, 0.20F);
    target.object_bounds = {vec3(), target.object_dimensions * 0.5F};
    target.state = interaction::ObjectState::Free;

    interaction::GraspAffordance affordance{};
    affordance.id = 7U;
    affordance.hand = hand;
    affordance.approach_direction_object = vec3(0.0F, 0.0F, 1.0F);
    affordance.clearance_radius = 0.04F;
    target.affordances.push_back(affordance);
    return target;
}

interaction::Transform make_reach_waypoint(
    const interaction::InteractionTarget& target) {
    const vec3 radial = quat_mul_vec3(
        target.object_world.rotation, vec3(0.0F, 0.0F, -0.40F));
    return {
        target.object_world.position + radial,
        target.object_world.rotation,
    };
}

interaction::PickEntryPreview ready_preview() {
    interaction::PickEntryPreview preview{};
    preview.path_feasible = true;
    preview.match_ready = true;
    return preview;
}

void require_slot_invariants(
    const interaction::PickEntrySlots& slots,
    const interaction::Transform& reach,
    const interaction::InteractionTarget& target) {
    require(
        slots.clearance_chord_m > 0.0F,
        "clearance chord must be positive");
    require(
        slots.preserved_standoff_m >= 0.35F &&
        slots.preserved_standoff_m <= 0.45F,
        "preserved standoff is outside the interaction envelope");

    for (size_t index = 0; index < slots.ordered.size(); ++index) {
        const interaction::PickEntrySlot& slot = slots.ordered[index];
        require(
            slot.identity == (index == 0U
                ? interaction::PickEntrySlotIdentity::Plus
                : interaction::PickEntrySlotIdentity::Minus),
            "slot identities are not in deterministic Plus/Minus order");
        require_near(
            planar_distance(slot.waypoint.position, target.object_world.position),
            slots.preserved_standoff_m,
            kTolerance,
            "slot did not preserve object standoff");
        require_near(
            planar_distance(slot.waypoint.position, reach.position),
            slots.clearance_chord_m,
            kTolerance,
            "slot did not move by the computed clearance chord");
        require_near(
            slot.waypoint.position.y,
            reach.position.y,
            kTolerance,
            "slot changed Reach height");
        require_near(
            quaternion_sign_distance(
                slot.waypoint.rotation, reach.rotation),
            0.0F,
            kTolerance,
            "slot changed Reach rotation");
        require_near(
            slot.prospective_root.world_x,
            slot.waypoint.position.x,
            0.0F,
            "prospective root x differs from slot waypoint");
        require_near(
            slot.prospective_root.world_z,
            slot.waypoint.position.z,
            0.0F,
            "prospective root z differs from slot waypoint");
    }
}

void test_geometry_preserves_standoff_and_clearance_chord() {
    const interaction::InteractionTarget target = make_target();
    const interaction::Transform reach = make_reach_waypoint(target);
    const interaction::PickEntrySlots slots =
        interaction::make_pick_entry_slots(reach, target);

    require_slot_invariants(slots, reach, target);
    require_near(
        slots.preserved_standoff_m,
        0.40F,
        kTolerance,
        "planner changed authored Reach standoff");
    require_near(
        slots.clearance_chord_m,
        0.141F,
        kTolerance,
        "planner changed support-radius clearance chord");
    require(
        std::string(interaction::pick_entry_slot_name(
            interaction::PickEntrySlotIdentity::Plus)) == "Plus",
        "Plus diagnostic name changed");
    require(
        std::string(interaction::pick_entry_slot_name(
            interaction::PickEntrySlotIdentity::Minus)) == "Minus",
        "Minus diagnostic name changed");
}

void test_rotated_translated_object_preserves_all_invariants() {
    const quat rotation = quat_from_angle_axis(
        0.73F, vec3(0.0F, 1.0F, 0.0F));
    const interaction::InteractionTarget target = make_target(
        interaction::Hand::Right,
        vec3(4.25F, 1.15F, -3.50F),
        rotation);
    const interaction::Transform reach = make_reach_waypoint(target);
    const interaction::PickEntrySlots slots =
        interaction::make_pick_entry_slots(reach, target);

    require_slot_invariants(slots, reach, target);
    require_near(
        slots.preserved_standoff_m,
        0.40F,
        kTolerance,
        "rotation or translation changed standoff");
    require_near(
        slots.clearance_chord_m,
        0.141F,
        kTolerance,
        "rotation or translation changed clearance chord");
}

void test_selection_filters_before_hand_preference() {
    const interaction::Transform right_reach =
        make_reach_waypoint(make_target(interaction::Hand::Right));
    const interaction::InteractionTarget right_target =
        make_target(interaction::Hand::Right);
    const interaction::PickEntrySlots right_slots =
        interaction::make_pick_entry_slots(right_reach, right_target);

    const interaction::InteractionTarget left_target =
        make_target(interaction::Hand::Left);
    const interaction::PickEntrySlots left_slots =
        interaction::make_pick_entry_slots(
            make_reach_waypoint(left_target), left_target);

    const std::array<interaction::PickEntryPreview, 2> both_ready{
        ready_preview(), ready_preview()};
    const std::array<bool, 2> both_permitted{true, true};
    const std::optional<size_t> right_choice =
        interaction::choose_pick_entry_slot(
            right_slots,
            both_ready,
            both_permitted,
            interaction::Hand::Right);
    const std::optional<size_t> left_choice =
        interaction::choose_pick_entry_slot(
            left_slots,
            both_ready,
            both_permitted,
            interaction::Hand::Left);
    require(right_choice.has_value(), "right hand did not choose a slot");
    require(left_choice.has_value(), "left hand did not choose a slot");
    require(
        *right_choice != *left_choice,
        "left and right hand did not reverse deterministic preference");
    require(
        right_slots.ordered[*right_choice].hand_score >
            right_slots.ordered[1U - *right_choice].hand_score,
        "right hand did not choose its higher hand score");
    require(
        left_slots.ordered[*left_choice].hand_score >
            left_slots.ordered[1U - *left_choice].hand_score,
        "left hand did not choose its higher hand score");

    const size_t lower_score = 1U - *right_choice;
    std::array<bool, 2> only_lower_permitted{false, false};
    only_lower_permitted[lower_score] = true;
    const std::optional<size_t> permitted_choice =
        interaction::choose_pick_entry_slot(
            right_slots,
            both_ready,
            only_lower_permitted,
            interaction::Hand::Right);
    require(
        permitted_choice == std::optional<size_t>(lower_score),
        "an ineligible higher-score slot won selection");

    std::array<interaction::PickEntryPreview, 2> one_ready = both_ready;
    one_ready[lower_score].path_feasible = false;
    const std::optional<size_t> one_eligible =
        interaction::choose_pick_entry_slot(
            right_slots,
            one_ready,
            both_permitted,
            interaction::Hand::Right);
    require(
        one_eligible == right_choice,
        "the only eligible preview did not win");

    const std::array<bool, 2> none_permitted{false, false};
    require(
        !interaction::choose_pick_entry_slot(
             right_slots,
             both_ready,
             none_permitted,
             interaction::Hand::Right).has_value(),
        "selection did not return nullopt when no slot was eligible");

    interaction::PickEntrySlots tied = right_slots;
    tied.ordered[0].hand_score = 0.0F;
    tied.ordered[1].hand_score = 0.0F;
    require(
        interaction::choose_pick_entry_slot(
            tied, both_ready, both_permitted, interaction::Hand::Right) ==
            std::optional<size_t>(0U),
        "right-hand tie break was not deterministic");
    require(
        interaction::choose_pick_entry_slot(
            tied, both_ready, both_permitted, interaction::Hand::Left) ==
            std::optional<size_t>(1U),
        "left-hand tie break was not deterministic");
}

interaction::Database make_waypoint_database() {
    interaction::Database database{};
    database.frame_count = 4U;
    database.bone_count = g1_skeleton::BoneCount;
    database.clip_count = 1U;
    database.range_starts = {0};
    database.range_stops = {4};
    database.phases = {
        static_cast<uint8_t>(interaction::Phase::Approach),
        static_cast<uint8_t>(interaction::Phase::Reach),
        static_cast<uint8_t>(interaction::Phase::Contact),
        static_cast<uint8_t>(interaction::Phase::Lift),
    };
    database.positions.assign(
        static_cast<size_t>(database.frame_count) * database.bone_count * 3U,
        0.0F);
    database.rotations.assign(
        static_cast<size_t>(database.frame_count) * database.bone_count * 4U,
        0.0F);
    for (size_t row = 0;
         row < static_cast<size_t>(database.frame_count) * database.bone_count;
         ++row) {
        database.rotations[row * 4U] = 1.0F;
    }
    database.object_positions.assign(
        static_cast<size_t>(database.frame_count) * 3U, 0.0F);
    database.object_rotations.assign(
        static_cast<size_t>(database.frame_count) * 4U, 0.0F);
    for (size_t frame = 0; frame < database.frame_count; ++frame) {
        database.object_rotations[frame * 4U] = 1.0F;
    }

    constexpr size_t reach = 1U;
    constexpr size_t root = g1_skeleton::Simulation;
    const size_t root_row = reach * database.bone_count + root;
    database.positions[root_row * 3U] = 1.0F;
    database.positions[root_row * 3U + 1U] = 0.25F;
    database.positions[root_row * 3U + 2U] = 1.60F;
    database.object_positions[reach * 3U] = 1.0F;
    database.object_positions[reach * 3U + 1U] = 0.75F;
    database.object_positions[reach * 3U + 2U] = 2.0F;
    return database;
}

void test_reach_waypoint_maps_clip_root_into_scene() {
    const interaction::Database database = make_waypoint_database();
    interaction::InteractionTarget target = make_target(
        interaction::Hand::Right,
        vec3(4.0F, 0.75F, -2.0F),
        quat_from_angle_axis(0.50F * PIf, vec3(0.0F, 1.0F, 0.0F)));
    const interaction::Transform actual =
        interaction::make_pick_reach_waypoint(database, target);
    const vec3 expected_offset = quat_mul_vec3(
        target.object_world.rotation, vec3(0.0F, 0.0F, -0.40F));
    require_near(
        actual.position,
        vec3(
            target.object_world.position.x + expected_offset.x,
            0.25F,
            target.object_world.position.z + expected_offset.z),
        kTolerance,
        "Reach root was not mapped into the target scene");
    require_near(
        quaternion_sign_distance(
            actual.rotation, target.object_world.rotation),
        0.0F,
        kTolerance,
        "Reach root rotation was not mapped into the target scene");
    require_throws(
        [&] { interaction::make_pick_reach_waypoint(database, target, 1U); },
        "out-of-range clip index was accepted");
}

void test_malformed_geometry_throws() {
    const interaction::InteractionTarget valid = make_target();
    const interaction::Transform reach = make_reach_waypoint(valid);

    interaction::InteractionTarget malformed_dimensions = valid;
    malformed_dimensions.object_dimensions.x = 0.0F;
    require_throws(
        [&] {
            interaction::make_pick_entry_slots(reach, malformed_dimensions);
        },
        "nonpositive object dimensions were accepted");

    interaction::InteractionTarget no_affordance = valid;
    no_affordance.affordances.clear();
    require_throws(
        [&] { interaction::make_pick_entry_slots(reach, no_affordance); },
        "missing affordance was accepted");

    interaction::InteractionTarget two_affordances = valid;
    two_affordances.affordances.push_back(valid.affordances.front());
    require_throws(
        [&] { interaction::make_pick_entry_slots(reach, two_affordances); },
        "multiple affordances were accepted");

    interaction::InteractionTarget malformed_object_rotation = valid;
    malformed_object_rotation.object_world.rotation = quat(0, 0, 0, 0);
    require_throws(
        [&] {
            interaction::make_pick_entry_slots(
                reach, malformed_object_rotation);
        },
        "malformed object rotation was accepted");

    interaction::Transform malformed_reach = reach;
    malformed_reach.rotation = quat(0, 0, 0, 0);
    require_throws(
        [&] { interaction::make_pick_entry_slots(malformed_reach, valid); },
        "malformed Reach rotation was accepted");
}

}  // namespace

int main() {
    try {
        test_geometry_preserves_standoff_and_clearance_chord();
        test_rotated_translated_object_preserves_all_invariants();
        test_selection_filters_before_hand_preference();
        test_reach_waypoint_maps_clip_root_into_scene();
        test_malformed_geometry_throws();
        std::cout << "interaction_pick_approach tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interaction_pick_approach test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
