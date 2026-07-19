#include "database.h"
#include "interaction_controller_adapter.h"
#include "interaction_database.h"
#include "interaction_pickup_provenance.h"
#include "interaction_smart_pickup_scenarios.h"

#include "interaction_pick_slots.h"
#include "interaction_smart_pickup_scene.h"
#include "locomotion_timing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kGeometryTolerance = 1.0e-5F;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

uint32_t float_bits(float value) {
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool same_float_bits(float left, float right) {
    return float_bits(left) == float_bits(right);
}

bool same_vec3_bits(vec3 left, vec3 right) {
    return same_float_bits(left.x, right.x) &&
        same_float_bits(left.y, right.y) &&
        same_float_bits(left.z, right.z);
}

bool same_quat_bits(quat left, quat right) {
    return same_float_bits(left.w, right.w) &&
        same_float_bits(left.x, right.x) &&
        same_float_bits(left.y, right.y) &&
        same_float_bits(left.z, right.z);
}

bool same_transform_bits(
    interaction::Transform left,
    interaction::Transform right) {
    return same_vec3_bits(left.position, right.position) &&
        same_quat_bits(left.rotation, right.rotation);
}

bool same_bounds_bits(
    interaction::ObjectLocalBounds left,
    interaction::ObjectLocalBounds right) {
    return same_vec3_bits(left.center_object, right.center_object) &&
        same_vec3_bits(
            left.half_extents_object, right.half_extents_object);
}

bool same_provenance_bits(
    const interaction::PickupSourceProvenance& left,
    const interaction::PickupSourceProvenance& right) {
    return left.clip_ordinal == right.clip_ordinal &&
        left.range_start == right.range_start &&
        left.range_stop == right.range_stop &&
        left.entry_global_frame == right.entry_global_frame &&
        left.contact_global_frame == right.contact_global_frame &&
        left.lift_global_frame == right.lift_global_frame &&
        left.hold_global_frame == right.hold_global_frame &&
        left.active_hand == right.active_hand &&
        left.object_profile_id == right.object_profile_id &&
        same_bounds_bits(left.object_bounds, right.object_bounds) &&
        same_transform_bits(left.hand_in_object, right.hand_in_object) &&
        same_float_bits(
            left.source_support_height_m,
            right.source_support_height_m);
}

template<class Exception, class Callback>
void require_throws(Callback&& callback, const std::string& label) {
    bool observed = false;
    try {
        std::forward<Callback>(callback)();
    } catch (const Exception&) {
        observed = true;
    }
    require(observed, label + " did not throw the required exception");
}

interaction::CertifiedPickupSourceLocalRow certified_local_row() {
    interaction::CertifiedPickupSourceLocalRow row{};
    row.clip_ordinal = 7;
    row.range_start = 100;
    row.range_stop = 180;
    row.entry_local_frame = 0;
    row.contact_local_frame = 5;
    row.lift_local_frame = 10;
    row.hold_local_frame = 15;
    row.reverse_start_local_frame = 79;
    row.active_hand = interaction::Hand::Right;
    row.object_profile_id = 9001U;
    row.object_bounds = {
        vec3(0.01F, -0.02F, 0.03F),
        vec3(0.04F, 0.05F, 0.06F),
    };
    row.hand_in_object = {
        vec3(0.07F, -0.08F, 0.09F),
        quat(0.5F, 0.5F, -0.5F, 0.5F),
    };
    row.source_support_height_m = 0.375F;
    row.sequence_id = "pickup_table__synthetic__001";
    row.object_id = "synthetic_object";
    row.join_key_sha256 =
        "0123456789abcdef0123456789abcdef"
        "0123456789abcdef0123456789abcdef";
    row.source_id = UINT64_C(0x0123456789abcdef);
    return row;
}

void test_checked_frame_conversions_and_local_row_factory() {
    using namespace interaction;
    require(checked_pickup_global_to_local_frame(100, 100, 180) == 0,
        "global range start did not convert to local zero");
    require(checked_pickup_global_to_local_frame(179, 100, 180) == 79,
        "global final frame did not convert to clip_length-1");
    require(checked_pickup_local_to_global_frame(0, 100, 180) == 100,
        "local zero did not convert to nonzero global range start");
    require(checked_pickup_local_to_global_frame(79, 100, 180) == 179,
        "local clip_length-1 did not convert to final global frame");
    for (int32_t local : {0, 5, 10, 15, 79}) {
        const int32_t global = checked_pickup_local_to_global_frame(
            local, 100, 180);
        require(checked_pickup_global_to_local_frame(
                    global, 100, 180) == local,
            "checked frame conversion did not round trip");
    }

    require_throws<std::out_of_range>([] {
        (void)checked_pickup_global_to_local_frame(-1, 100, 180);
    }, "negative global frame");
    require_throws<std::out_of_range>([] {
        (void)checked_pickup_global_to_local_frame(180, 100, 180);
    }, "global range stop");
    require_throws<std::out_of_range>([] {
        (void)checked_pickup_local_to_global_frame(-1, 100, 180);
    }, "negative local frame");
    require_throws<std::out_of_range>([] {
        (void)checked_pickup_local_to_global_frame(80, 100, 180);
    }, "local clip length");
    require_throws<std::invalid_argument>([] {
        (void)checked_pickup_global_to_local_frame(100, 100, 100);
    }, "empty global range");
    require_throws<std::invalid_argument>([] {
        (void)checked_pickup_local_to_global_frame(0, 180, 100);
    }, "reversed local range");
    require_throws<std::invalid_argument>([] {
        (void)checked_pickup_local_to_global_frame(0, -1, 100);
    }, "negative range start");
    require_throws<std::overflow_error>([] {
        (void)checked_pickup_local_to_global_frame(
            std::numeric_limits<int32_t>::max(),
            1,
            std::numeric_limits<int32_t>::max());
    }, "checked local-to-global addition overflow");

    const CertifiedPickupSourceLocalRow local = certified_local_row();
    const CertifiedPickupSourceIdentity identity =
        make_certified_pickup_source_identity(local);
    require(identity.provenance.clip_ordinal == local.clip_ordinal &&
            identity.provenance.range_start == local.range_start &&
            identity.provenance.range_stop == local.range_stop &&
            identity.provenance.entry_global_frame == 100 &&
            identity.provenance.contact_global_frame == 105 &&
            identity.provenance.lift_global_frame == 110 &&
            identity.provenance.hold_global_frame == 115 &&
            identity.reverse_start_global_frame == 179 &&
            identity.provenance.active_hand == local.active_hand &&
            identity.provenance.object_profile_id ==
                local.object_profile_id &&
            same_bounds_bits(
                identity.provenance.object_bounds, local.object_bounds) &&
            same_transform_bits(
                identity.provenance.hand_in_object,
                local.hand_in_object) &&
            same_float_bits(
                identity.provenance.source_support_height_m,
                local.source_support_height_m) &&
            identity.sequence_id == local.sequence_id &&
            identity.object_id == local.object_id &&
            identity.join_key_sha256 == local.join_key_sha256 &&
            identity.source_id == local.source_id,
        "local-row factory did not copy and convert every field exactly");

    CertifiedPickupSourceLocalRow invalid = local;
    invalid.contact_local_frame = invalid.entry_local_frame;
    require_throws<std::invalid_argument>([&] {
        (void)make_certified_pickup_source_identity(invalid);
    }, "unordered local pickup events");
    invalid = local;
    invalid.reverse_start_local_frame = 80;
    require_throws<std::out_of_range>([&] {
        (void)make_certified_pickup_source_identity(invalid);
    }, "local reverse start at clip length");
}

void test_certified_registry_is_an_exact_raw_provenance_join() {
    using namespace interaction;
    const CertifiedPickupSourceIdentity identity =
        make_certified_pickup_source_identity(certified_local_row());
    const CertifiedPickupSourceRegistry registry({identity});
    const std::optional<CertifiedPickupSourceIdentity> resolved =
        registry.resolve_exact(identity.provenance);
    require(resolved.has_value() &&
            same_provenance_bits(
                resolved->provenance, identity.provenance) &&
            resolved->sequence_id == identity.sequence_id &&
            resolved->object_id == identity.object_id &&
            resolved->reverse_start_global_frame ==
                identity.reverse_start_global_frame &&
            resolved->join_key_sha256 == identity.join_key_sha256 &&
            resolved->source_id == identity.source_id,
        "certified registry did not return the exact global row");

    const auto rejects = [&](const auto& mutate, const std::string& label) {
        PickupSourceProvenance changed = identity.provenance;
        mutate(changed);
        require(!registry.resolve_exact(changed).has_value(),
            "registry accepted mutated " + label);
    };
    rejects([](PickupSourceProvenance& value) { ++value.clip_ordinal; },
        "clip ordinal");
    rejects([](PickupSourceProvenance& value) { --value.range_start; },
        "range start");
    rejects([](PickupSourceProvenance& value) { ++value.range_stop; },
        "range stop");
    rejects([](PickupSourceProvenance& value) {
        ++value.entry_global_frame;
    }, "entry event");
    rejects([](PickupSourceProvenance& value) {
        ++value.contact_global_frame;
    }, "contact event");
    rejects([](PickupSourceProvenance& value) {
        ++value.lift_global_frame;
    }, "lift event");
    rejects([](PickupSourceProvenance& value) {
        ++value.hold_global_frame;
    }, "hold event");
    rejects([](PickupSourceProvenance& value) {
        value.active_hand = Hand::Left;
    }, "active hand");
    rejects([](PickupSourceProvenance& value) {
        ++value.object_profile_id;
    }, "object profile");
    rejects([](PickupSourceProvenance& value) {
        value.object_bounds.half_extents_object.x = std::nextafter(
            value.object_bounds.half_extents_object.x,
            std::numeric_limits<float>::infinity());
    }, "object bounds float32 component");
    rejects([](PickupSourceProvenance& value) {
        value.hand_in_object.position.y = std::nextafter(
            value.hand_in_object.position.y,
            std::numeric_limits<float>::infinity());
    }, "grasp translation float32 component");
    rejects([](PickupSourceProvenance& value) {
        value.hand_in_object.rotation = -value.hand_in_object.rotation;
    }, "grasp quaternion sign");
    rejects([](PickupSourceProvenance& value) {
        value.source_support_height_m = std::nextafter(
            value.source_support_height_m,
            std::numeric_limits<float>::infinity());
    }, "source support height float32 component");

    require_throws<std::invalid_argument>([&] {
        (void)CertifiedPickupSourceRegistry({identity, identity});
    }, "duplicate raw provenance");
    const auto rejects_row = [&](const auto& mutate, const std::string& label) {
        CertifiedPickupSourceIdentity changed = identity;
        mutate(changed);
        require_throws<std::invalid_argument>([&] {
            (void)CertifiedPickupSourceRegistry({changed});
        }, label);
    };
    rejects_row([](CertifiedPickupSourceIdentity& value) {
        value.sequence_id.clear();
    }, "empty certified sequence identity");
    rejects_row([](CertifiedPickupSourceIdentity& value) {
        value.object_id.clear();
    }, "empty certified object identity");
    rejects_row([](CertifiedPickupSourceIdentity& value) {
        value.reverse_start_global_frame = value.provenance.range_stop;
    }, "out-of-range certified reverse start");
    rejects_row([](CertifiedPickupSourceIdentity& value) {
        value.join_key_sha256[0] = 'A';
    }, "noncanonical certified join SHA");
    rejects_row([](CertifiedPickupSourceIdentity& value) {
        ++value.source_id;
    }, "certified source ID not derived from join SHA");
}

bool finite_transform(const interaction::Transform& value) {
    return std::isfinite(value.position.x) &&
        std::isfinite(value.position.y) &&
        std::isfinite(value.position.z) &&
        std::isfinite(value.rotation.w) &&
        std::isfinite(value.rotation.x) &&
        std::isfinite(value.rotation.y) &&
        std::isfinite(value.rotation.z);
}

float planar_distance(
    const interaction::Transform& left,
    const interaction::Transform& right) {
    return std::hypot(
        left.position.x - right.position.x,
        left.position.z - right.position.z);
}

float planar_yaw(const interaction::Transform& value) {
    const vec3 forward = quat_mul_vec3(
        value.rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

float angular_distance(float left, float right) {
    return std::fabs(std::remainder(left - right, 2.0F * kPi));
}

bool same_local_target_data(
    const interaction::InteractionTarget& left,
    const interaction::InteractionTarget& right) {
    if (left.handle != right.handle ||
        left.object_profile_id != right.object_profile_id ||
        left.state != right.state ||
        left.owner_request != right.owner_request ||
        !same_vec3_bits(left.object_bounds.center_object,
            right.object_bounds.center_object) ||
        !same_vec3_bits(left.object_bounds.half_extents_object,
            right.object_bounds.half_extents_object) ||
        !same_vec3_bits(left.object_dimensions, right.object_dimensions) ||
        !same_vec3_bits(left.table_size, right.table_size) ||
        left.affordances.size() != right.affordances.size()) {
        return false;
    }
    for (size_t index = 0U; index < left.affordances.size(); ++index) {
        const interaction::GraspAffordance& expected = left.affordances[index];
        const interaction::GraspAffordance& actual = right.affordances[index];
        if (expected.id != actual.id || expected.hand != actual.hand ||
            !same_transform_bits(expected.hand_in_object,
                actual.hand_in_object) ||
            !same_vec3_bits(expected.approach_direction_object,
                actual.approach_direction_object) ||
            !same_float_bits(expected.clearance_radius,
                actual.clearance_radius) ||
            expected.interaction_slots.size() !=
                actual.interaction_slots.size()) {
            return false;
        }
        for (size_t slot = 0U;
             slot < expected.interaction_slots.size();
             ++slot) {
            const interaction::GraspInteractionSlot& expected_slot =
                expected.interaction_slots[slot];
            const interaction::GraspInteractionSlot& actual_slot =
                actual.interaction_slots[slot];
            if (expected_slot.id != actual_slot.id ||
                !same_float_bits(expected_slot.root_x_object_m,
                    actual_slot.root_x_object_m) ||
                !same_float_bits(expected_slot.root_z_object_m,
                    actual_slot.root_z_object_m) ||
                !same_float_bits(expected_slot.root_yaw_object_radians,
                    actual_slot.root_yaw_object_radians)) {
                return false;
            }
        }
    }
    return true;
}

bool same_local_destination_data(
    const interaction::PlacementSurface& left,
    const interaction::PlacementSurface& right) {
    if (!(left.handle == right.handle) ||
        !same_vec3_bits(left.support_volume_size,
            right.support_volume_size) ||
        !same_float_bits(left.half_extent_x_m, right.half_extent_x_m) ||
        !same_float_bits(left.half_extent_z_m, right.half_extent_z_m) ||
        !same_float_bits(left.overhead_clearance_m,
            right.overhead_clearance_m) ||
        left.affordances.size() != right.affordances.size()) {
        return false;
    }
    for (size_t index = 0U; index < left.affordances.size(); ++index) {
        const interaction::PlaceAffordance& expected = left.affordances[index];
        const interaction::PlaceAffordance& actual = right.affordances[index];
        if (expected.id != actual.id ||
            !same_transform_bits(expected.object_in_surface,
                actual.object_in_surface) ||
            !same_vec3_bits(expected.support_point_object,
                actual.support_point_object) ||
            !same_vec3_bits(expected.approach_direction_surface,
                actual.approach_direction_surface) ||
            !same_float_bits(expected.clearance_radius,
                actual.clearance_radius)) {
            return false;
        }
    }
    return true;
}

interaction::InteractionTarget materialize_target(
    const interaction::InteractionTarget& authored,
    const interaction::SmartPickupDemoScenario& scenario) {
    interaction::InteractionTarget target = authored;
    target.table_world = interaction::compose(
        scenario.scene_from_authored, authored.table_world);
    target.object_world = interaction::compose(
        scenario.scene_from_authored, authored.object_world);
    return target;
}

interaction::PlacementSurface materialize_destination(
    const interaction::PlacementSurface& authored,
    const interaction::SmartPickupDemoScenario& scenario) {
    interaction::PlacementSurface destination = authored;
    destination.surface_world = interaction::compose(
        scenario.scene_from_authored, authored.surface_world);
    destination.support_volume_world = interaction::compose(
        scenario.scene_from_authored, authored.support_volume_world);
    return destination;
}

const interaction::SmartPickupDemoScenario& scenario_named(
    const std::string_view id) {
    const auto& scenarios = interaction::smart_pickup_demo_scenarios();
    const auto found = std::find_if(
        scenarios.begin(), scenarios.end(),
        [&](const interaction::SmartPickupDemoScenario& scenario) {
            return std::string_view(scenario.id) == id;
        });
    require(found != scenarios.end(), "missing Smart Pickup scenario " +
        std::string(id));
    return *found;
}

uint32_t selected_slot_id(
    const interaction::PickSlotSelection& selection,
    const std::string& label) {
    require(selection.selected_index.has_value(),
        label + " did not select a slot");
    require(*selection.selected_index < selection.ordered.size(),
        label + " selected index is outside diagnostics");
    return selection.ordered[*selection.selected_index].id;
}

interaction::MappedPickSlot mapped_slot_named(
    const interaction::PickSlotSelection& selection,
    uint32_t slot_id,
    const std::string& label) {
    const auto found = std::find_if(
        selection.ordered.begin(), selection.ordered.end(),
        [&](const interaction::MappedPickSlot& slot) {
            return slot.id == slot_id;
        });
    require(found != selection.ordered.end(),
        label + " is missing mapped slot " + std::to_string(slot_id));
    return *found;
}

void test_catalog_api_order_and_geometry_contract() {
    using namespace interaction;
    constexpr std::array<std::string_view, 7> expected_ids{{
        "clear_front",
        "clear_left",
        "clear_right",
        "translated",
        "yawed",
        "alternate_slot",
        "all_blocked",
    }};
    constexpr std::array<SmartPickupScenarioClass, 7> expected_classes{{
        SmartPickupScenarioClass::Clear,
        SmartPickupScenarioClass::Clear,
        SmartPickupScenarioClass::Clear,
        SmartPickupScenarioClass::Clear,
        SmartPickupScenarioClass::Clear,
        SmartPickupScenarioClass::AlternateSlot,
        SmartPickupScenarioClass::AllBlocked,
    }};
    const auto& scenarios = smart_pickup_demo_scenarios();
    require(kSmartPickupDemoRepeatCount == 10U,
        "Smart Pickup catalog must expose exactly ten repeats");
    require(
        static_cast<uint8_t>(SmartPickupScenarioClass::Clear) !=
                static_cast<uint8_t>(SmartPickupScenarioClass::AlternateSlot) &&
            static_cast<uint8_t>(SmartPickupScenarioClass::Clear) !=
                static_cast<uint8_t>(SmartPickupScenarioClass::AllBlocked) &&
            static_cast<uint8_t>(SmartPickupScenarioClass::AlternateSlot) !=
                static_cast<uint8_t>(SmartPickupScenarioClass::AllBlocked),
        "Smart Pickup scenario classes must remain distinct");
    require(scenarios.size() == expected_ids.size(),
        "Smart Pickup catalog must contain exactly seven cases");
    for (size_t index = 0U; index < scenarios.size(); ++index) {
        const SmartPickupDemoScenario& scenario = scenarios[index];
        const std::string_view id = scenario.id;
        const std::vector<PickNavigationObstacle>& obstacles =
            scenario.obstacles;
        require(id == expected_ids[index],
            "Smart Pickup scenario ID/order changed at index " +
                std::to_string(index));
        require(scenario.expected_class == expected_classes[index],
            std::string(id) + " has the wrong expected class");
        require(finite_transform(scenario.initial_root_world) &&
                finite_transform(scenario.scene_from_authored),
            std::string(id) + " contains a non-finite transform");
        require(std::fabs(scenario.initial_root_world.position.y) <=
                kGeometryTolerance,
            std::string(id) + " initial root is not on Y=0");
        require(std::fabs(scenario.scene_from_authored.position.y) <=
                kGeometryTolerance &&
                std::fabs(scenario.scene_from_authored.rotation.x) <=
                    kGeometryTolerance &&
                std::fabs(scenario.scene_from_authored.rotation.z) <=
                    kGeometryTolerance,
            std::string(id) + " scene transform is not planar");
        for (const PickNavigationObstacle& obstacle : obstacles) {
            require(std::isfinite(obstacle.center_world.x) &&
                    std::isfinite(obstacle.center_world.y) &&
                    std::isfinite(obstacle.center_world.z) &&
                    std::isfinite(obstacle.size_world.x) &&
                    std::isfinite(obstacle.size_world.y) &&
                    std::isfinite(obstacle.size_world.z) &&
                    obstacle.size_world.x > 0.0F &&
                    obstacle.size_world.y > 0.0F &&
                    obstacle.size_world.z > 0.0F,
                std::string(id) + " has an invalid authored blocker");
        }
    }
    for (size_t left = 0U; left < 3U; ++left) {
        for (size_t right = left + 1U; right < 3U; ++right) {
            require(planar_distance(
                        scenarios[left].initial_root_world,
                        scenarios[right].initial_root_world) >= 0.30F,
                "clear starts are not separated by at least 0.30 m");
        }
    }
    float maximum_yaw_span = 0.0F;
    for (size_t left = 0U; left < 3U; ++left) {
        for (size_t right = left + 1U; right < 3U; ++right) {
            maximum_yaw_span = std::max(maximum_yaw_span, angular_distance(
                planar_yaw(scenarios[left].initial_root_world),
                planar_yaw(scenarios[right].initial_root_world)));
        }
    }
    require(maximum_yaw_span >= 0.25F * kPi,
        "clear starts do not span at least 45 degrees of initial yaw");
}

void test_materialized_scenes_and_selector_outcomes() {
    using namespace interaction;
    const InteractionTarget authored = make_smart_pickup_demo_target();
    const PlacementSurface authored_destination =
        make_smart_pickup_demo_destination_surface(authored);
    std::set<uint32_t> clear_slot_ids;

    for (const SmartPickupDemoScenario& scenario :
         smart_pickup_demo_scenarios()) {
        const std::string id = std::string(std::string_view(scenario.id));
        const InteractionTarget target = materialize_target(authored, scenario);
        const PlacementSurface destination =
            materialize_destination(authored_destination, scenario);
        require(same_local_target_data(authored, target),
            id + " changed bit-exact authored target-local data");
        require(same_local_destination_data(authored_destination, destination),
            id + " changed bit-exact authored destination-local data");
        require(target.affordances.size() == 1U,
            id + " materialized an invalid affordance count");
        const PickSlotSelection selected = select_pick_slot(
            scenario.initial_root_world,
            target,
            target.affordances.front(),
            scenario.obstacles);

        if (scenario.expected_class == SmartPickupScenarioClass::Clear) {
            require(selected.reason == PickSlotReason::None,
                id + " clear case did not report PickSlotReason::None");
            clear_slot_ids.insert(selected_slot_id(selected, id));
        } else if (scenario.expected_class ==
                   SmartPickupScenarioClass::AlternateSlot) {
            require(!scenario.obstacles.empty(),
                "alternate_slot has no authored blocker");
            const PickSlotSelection nominal = select_pick_slot(
                scenario.initial_root_world,
                target,
                target.affordances.front(),
                {});
            const uint32_t nominal_id = selected_slot_id(
                nominal, "alternate_slot nominal route");
            const uint32_t alternate_id = selected_slot_id(
                selected, "alternate_slot blocked route");
            require(selected.reason == PickSlotReason::None &&
                    nominal_id != alternate_id,
                "alternate_slot blocker did not change the nominal slot");
        } else if (scenario.expected_class ==
                   SmartPickupScenarioClass::AllBlocked) {
            require(!scenario.obstacles.empty() &&
                    !selected.selected_index.has_value() &&
                    selected.reason == PickSlotReason::AllSlotsBlocked,
                "all_blocked did not report typed AllSlotsBlocked failure");
        } else {
            require(false, id + " has an unknown expected class");
        }

        if (id == "translated" || id == "yawed") {
            require(!same_transform_bits(target.object_world,
                        authored.object_world) &&
                    !same_transform_bits(target.table_world,
                        authored.table_world) &&
                    !same_transform_bits(destination.surface_world,
                        authored_destination.surface_world) &&
                    !same_transform_bits(destination.support_volume_world,
                        authored_destination.support_volume_world),
                id + " did not change target and destination world geometry");
            const uint32_t transformed_slot_id = selected_slot_id(selected, id);
            const PickSlotSelection authored_mapping = select_pick_slot(
                scenario.initial_root_world,
                authored,
                authored.affordances.front(),
                {});
            const MappedPickSlot& authored_slot = mapped_slot_named(
                authored_mapping, transformed_slot_id, id + " authored map");
            const MappedPickSlot& transformed_slot = mapped_slot_named(
                selected, transformed_slot_id, id + " transformed map");
            require(!same_transform_bits(
                        authored_slot.root_world,
                        transformed_slot.root_world),
                id + " did not change the selected slot's world mapping");
        }
    }
    require(clear_slot_ids.size() >= 2U,
        "clear scenario matrix must exercise at least two selected slot IDs");

    const SmartPickupDemoScenario& translated = scenario_named("translated");
    require(std::hypot(translated.scene_from_authored.position.x,
                translated.scene_from_authored.position.z) >
            kGeometryTolerance,
        "translated scenario has no planar translation");
    const SmartPickupDemoScenario& yawed = scenario_named("yawed");
    require(std::fabs(planar_yaw(yawed.scene_from_authored)) >
            kGeometryTolerance,
        "yawed scenario has no world yaw");
}

void test_typed_target_compatibility_policy() {
    using namespace interaction;
    const InteractionTarget authored = make_smart_pickup_demo_target();
    require(authored.affordances.size() == 1U,
        "authored Smart Pickup target has no grasp affordance");
    const uint32_t affordance_id = authored.affordances.front().id;

    InteractionTarget compatible = authored;
    compatible.handle = {901U, 7U};
    compatible.object_profile_id = 991U;
    compatible.object_dimensions = 0.75F * authored.object_dimensions;
    compatible.object_bounds.half_extents_object =
        0.75F * authored.object_bounds.half_extents_object;
    require(classify_smart_pickup_target(compatible, affordance_id) ==
                SmartPickupTargetCompatibility::Compatible,
        "distinct object with copied Right-hand slots was rejected");
    const SmartPickupDemoScenario& clear = scenario_named("clear_front");
    const PickSlotSelection compatible_selection = select_pick_slot(
        clear.initial_root_world,
        compatible,
        compatible.affordances.front(),
        clear.obstacles);
    require(compatible_selection.selected_index.has_value(),
        "compatible distinct object did not map/select copied slots");

    require(classify_smart_pickup_target(
                compatible, affordance_id + 1000U) ==
                SmartPickupTargetCompatibility::MissingAffordance,
        "missing affordance ID did not report MissingAffordance");
    InteractionTarget unsupported = compatible;
    unsupported.affordances.front().hand = Hand::Left;
    require(classify_smart_pickup_target(unsupported, affordance_id) ==
                SmartPickupTargetCompatibility::UnsupportedHand,
        "Left-hand target did not report UnsupportedHand");
    InteractionTarget empty_slots = compatible;
    empty_slots.affordances.front().interaction_slots.clear();
    require(classify_smart_pickup_target(empty_slots, affordance_id) ==
                SmartPickupTargetCompatibility::NoAuthoredSlot,
        "empty authored slots did not report NoAuthoredSlot");

    constexpr std::array<SmartPickupTargetCompatibility, 4> values{{
        SmartPickupTargetCompatibility::Compatible,
        SmartPickupTargetCompatibility::MissingAffordance,
        SmartPickupTargetCompatibility::UnsupportedHand,
        SmartPickupTargetCompatibility::NoAuthoredSlot,
    }};
    constexpr std::array<std::string_view, 4> names{{
        "Compatible",
        "MissingAffordance",
        "UnsupportedHand",
        "NoAuthoredSlot",
    }};
    for (size_t index = 0U; index < values.size(); ++index) {
        const char* name = smart_pickup_target_compatibility_name(values[index]);
        require(name != nullptr && std::string_view(name) == names[index],
            "Smart Pickup compatibility name is unstable");
    }
}

interaction::FlatControllerPose flat_pose_at(
    const database& flat_database,
    int frame) {
    interaction::FlatControllerPose pose{};
    for (size_t bone = 0U;
         bone < interaction::kFlatControllerBoneCount;
         ++bone) {
        const int index = static_cast<int>(bone);
        pose.positions[bone] = flat_database.bone_positions(frame, index);
        pose.velocities[bone] = flat_database.bone_velocities(frame, index);
        pose.rotations[bone] = flat_database.bone_rotations(frame, index);
        pose.angular_velocities[bone] =
            flat_database.bone_angular_velocities(frame, index);
    }
    pose.foot_contacts[0] =
        flat_database.contact_states(frame, 0) ? 1U : 0U;
    pose.foot_contacts[1] =
        flat_database.contact_states(frame, 1) ? 1U : 0U;
    return pose;
}

interaction::LocomotionSnapshot published_flat_snapshot(
    const std::filesystem::path& flat_database_path,
    const interaction::Database& interaction_database) {
    database flat_database{};
    const std::string filename = flat_database_path.string();
    database_load(flat_database, filename.c_str());
    require(flat_database.nranges() > 0 && flat_database.nframes() > 0,
        "ordinary flat database is empty");
    const int frame = flat_database.range_starts(0);
    const int clip_stop = flat_database.range_stops(0);
    require(frame >= 0 && frame < clip_stop,
        "ordinary flat reference frame is invalid");

    interaction::FlatControllerPose flat_reference =
        flat_pose_at(flat_database, frame);
    flat_reference.velocities.fill(vec3());
    flat_reference.angular_velocities.fill(vec3());
    flat_reference.foot_contacts = {};
    require(!interaction_database.range_starts.empty(),
        "interaction database has no reference frame");
    interaction::Pose interaction_reference = interaction::pose_at_frame(
        interaction_database, interaction_database.range_starts.front());
    interaction_reference.velocities.fill(vec3());
    interaction_reference.angular_velocities.fill(vec3());
    interaction_reference.hand_dof = interaction::kFlatControllerRestHandDof;
    interaction_reference.hand_dof_velocities =
        interaction::kFlatControllerRestHandDofVelocities;
    interaction_reference.foot_contacts = {};
    interaction_reference.positions[g1_skeleton::Simulation] =
        flat_reference.positions[0];
    interaction_reference.velocities[g1_skeleton::Simulation] =
        flat_reference.velocities[0];
    interaction_reference.rotations[g1_skeleton::Simulation] =
        flat_reference.rotations[0];
    interaction_reference.angular_velocities[g1_skeleton::Simulation] =
        flat_reference.angular_velocities[0];

    interaction::LocomotionSnapshot snapshot{};
    snapshot.pose = interaction::expand_flat_controller_pose(
        flat_pose_at(flat_database, frame),
        interaction_reference,
        flat_reference);
    for (size_t index = 0U;
         index < locomotion_timing::kTrajectoryFrameOffsets.size();
         ++index) {
        const int future = frame +
            locomotion_timing::kTrajectoryFrameOffsets[index];
        require(future < clip_stop,
            "ordinary flat snapshot crosses its source clip");
        snapshot.future_root_positions[index] =
            flat_database.bone_positions(future, 0);
        snapshot.future_root_rotations[index] =
            flat_database.bone_rotations(future, 0);
    }
    return snapshot;
}

void test_full_pack_lifecycle_requires_actual_attachment(
    const std::filesystem::path& flat_database_path,
    const std::filesystem::path& full_pack_path) {
    using namespace interaction;
    const Database database = load_database(
        full_pack_path / "interaction_database.bin");
    const Features features = load_features(
        full_pack_path / "interaction_features.bin");
    validate_controller_interaction_pack(database, features);
    const LocomotionSnapshot flat = published_flat_snapshot(
        flat_database_path, database);
    const SmartPickupLifecycleWitness witness =
        run_smart_pickup_lifecycle(
            database,
            features,
            flat,
            scenario_named("clear_front"));

    require(witness.request_count == 1U,
        "full-pack lifecycle did not submit exactly one request");
    require(witness.attachment_edges == 1U,
        "full-pack lifecycle did not observe exactly one attach edge");
    require(witness.release_edges == 0U,
        "full-pack lifecycle unexpectedly released the pickup");
    require(witness.terminal_state == RuntimeState::Carry,
        "full-pack lifecycle did not terminate in Carry");
    require(witness.terminal_object_state == ObjectState::Held,
        "full-pack lifecycle target did not terminate Held");
    require(witness.attached,
        "full-pack lifecycle did not remain attached");
    require(witness.owner_request == witness.request_id,
        "full-pack lifecycle owner differs from its request");
    require(witness.target_after == witness.target_before,
        "full-pack lifecycle changed the target generation");
    require(witness.post_carry_tick_observed,
        "full-pack lifecycle stopped before a normal Carry tick");
    require(witness.grasp_preserved,
        "normal Carry tick did not preserve the frozen grasp");
    require(witness.carry_root_displacement_m > 0.0F,
        "normal Carry tick did not move the root");
    require(witness.carry_object_displacement_m > 0.0F,
        "normal Carry tick did not move the attached object");
    require(witness.actual_source.provenance.clip_ordinal >= 0,
        "runtime did not freeze the selected clip ordinal");
    require(witness.actual_source.provenance.range_start <=
            witness.actual_source.provenance.entry_global_frame,
        "runtime entry provenance precedes its global clip range");
    require(witness.actual_source.provenance.entry_global_frame <
            witness.actual_source.provenance.contact_global_frame,
        "runtime entry/contact provenance is unordered");
    require(witness.actual_source.provenance.contact_global_frame <
            witness.actual_source.provenance.lift_global_frame,
        "runtime contact/lift provenance is unordered");
    require(witness.actual_source.provenance.lift_global_frame <
            witness.actual_source.provenance.hold_global_frame,
        "runtime lift/hold provenance is unordered");
    require(witness.actual_source.provenance.hold_global_frame <
            witness.actual_source.provenance.range_stop,
        "runtime hold provenance exceeds its global clip range");
    require(std::isfinite(
                witness.actual_source.provenance.source_support_height_m),
        "runtime source support height is not finite");
    require(witness.actual_source.sequence_id.empty(),
        "uncertified full-pack lifecycle invented a sequence identity");
    require(witness.actual_source.object_id.empty(),
        "uncertified full-pack lifecycle invented an object identity");
    require(witness.actual_source.reverse_start_global_frame == -1,
        "uncertified full-pack lifecycle invented reverse-start identity");
    require(witness.actual_source.source_id == 0U,
        "uncertified full-pack lifecycle invented a source ID");
    require(witness.actual_source.join_key_sha256.empty(),
        "uncertified full-pack lifecycle invented a join SHA");
}

std::string f32_hex(float value) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::nouppercase
           << std::setfill('0') << std::setw(8) << float_bits(value);
    return stream.str();
}

void write_json_string(std::ostream& stream, const std::string& value) {
    stream << std::quoted(value);
}

void write_f32_hex_array(
    std::ostream& stream,
    std::initializer_list<float> values) {
    stream << '[';
    bool first = true;
    for (float value : values) {
        if (!first) stream << ',';
        first = false;
        write_json_string(stream, f32_hex(value));
    }
    stream << ']';
}

const interaction::SmartPickupSlotProvenance& selected_slot_provenance(
    const interaction::SmartPickupDemoScenario& scenario) {
    using namespace interaction;
    const InteractionTarget authored = make_smart_pickup_demo_target();
    const InteractionTarget target = materialize_target(authored, scenario);
    require(target.affordances.size() == 1U,
        std::string(scenario.id) + " has no unique affordance");
    const PickSlotSelection selection = select_pick_slot(
        scenario.initial_root_world,
        target,
        target.affordances.front(),
        scenario.obstacles);
    const uint32_t slot_id = selected_slot_id(
        selection, std::string(scenario.id));
    const auto& provenances = smart_pickup_demo_slot_provenance();
    const auto found = std::find_if(
        provenances.begin(), provenances.end(),
        [slot_id](const SmartPickupSlotProvenance& provenance) {
            return provenance.slot_id == slot_id;
        });
    require(found != provenances.end(),
        std::string(scenario.id) + " selected an unbaked slot");
    return *found;
}

void write_case_result(
    std::ostream& stream,
    const interaction::SmartPickupDemoScenario& scenario,
    uint32_t repeat,
    const interaction::SmartPickupSlotProvenance& slot,
    const interaction::SmartPickupLifecycleWitness& witness) {
    using namespace interaction;
    const PickupSourceProvenance& source = witness.actual_source.provenance;
    require(witness.terminal_state == RuntimeState::Carry,
        std::string(scenario.id) + " repeat " + std::to_string(repeat) +
            " oracle row did not end in Carry: state=" +
            std::to_string(static_cast<unsigned>(witness.terminal_state)) +
            " requests=" + std::to_string(witness.request_count) +
            " attach_edges=" + std::to_string(witness.attachment_edges) +
            " release_edges=" + std::to_string(witness.release_edges) +
            " attached=" + (witness.attached ? "true" : "false"));
    require(witness.terminal_object_state == ObjectState::Held,
        std::string(scenario.id) + " oracle target did not end Held");

    stream << '{';
    stream << "\"attached\":" << (witness.attached ? "true" : "false");
    stream << ",\"attachment_edges\":" << witness.attachment_edges;
    stream << ",\"case_id\":";
    write_json_string(stream, scenario.id);
    stream << ",\"carry_object_displacement_m\":"
           << witness.carry_object_displacement_m;
    stream << ",\"carry_root_displacement_m\":"
           << witness.carry_root_displacement_m;
    stream << ",\"grasp_preserved\":"
           << (witness.grasp_preserved ? "true" : "false");
    stream << ",\"owner_request\":" << witness.owner_request;
    stream << ",\"post_carry_tick_observed\":"
           << (witness.post_carry_tick_observed ? "true" : "false");
    stream << ",\"record_type\":\"case_result\"";
    stream << ",\"release_edges\":" << witness.release_edges;
    stream << ",\"repeat\":" << repeat;
    stream << ",\"request_count\":" << witness.request_count;
    stream << ",\"request_id\":" << witness.request_id;
    stream << ",\"selected_slot_contact_local_frame\":"
           << slot.contact_local_frame;
    stream << ",\"selected_slot_entry_local_frame\":"
           << slot.entry_local_frame;
    stream << ",\"selected_slot_id\":" << slot.slot_id;
    stream << ",\"selected_slot_sequence_id\":";
    write_json_string(stream, slot.sequence_id);
    stream << ",\"source_active_hand\":"
           << static_cast<unsigned>(source.active_hand);
    stream << ",\"source_clip_ordinal\":" << source.clip_ordinal;
    stream << ",\"source_contact_global_frame\":"
           << source.contact_global_frame;
    stream << ",\"source_entry_global_frame\":"
           << source.entry_global_frame;
    stream << ",\"source_hand_in_object_f32_hex\":";
    write_f32_hex_array(stream, {
        source.hand_in_object.position.x,
        source.hand_in_object.position.y,
        source.hand_in_object.position.z,
        source.hand_in_object.rotation.w,
        source.hand_in_object.rotation.x,
        source.hand_in_object.rotation.y,
        source.hand_in_object.rotation.z,
    });
    stream << ",\"source_hold_global_frame\":"
           << source.hold_global_frame;
    stream << ",\"source_id\":" << witness.actual_source.source_id;
    stream << ",\"source_join_key_sha256\":";
    write_json_string(stream, witness.actual_source.join_key_sha256);
    stream << ",\"source_lift_global_frame\":"
           << source.lift_global_frame;
    stream << ",\"source_object_bounds_f32_hex\":";
    write_f32_hex_array(stream, {
        source.object_bounds.center_object.x,
        source.object_bounds.center_object.y,
        source.object_bounds.center_object.z,
        source.object_bounds.half_extents_object.x,
        source.object_bounds.half_extents_object.y,
        source.object_bounds.half_extents_object.z,
    });
    stream << ",\"source_object_id\":";
    write_json_string(stream, witness.actual_source.object_id);
    stream << ",\"source_object_profile_id\":"
           << source.object_profile_id;
    stream << ",\"source_range_start\":" << source.range_start;
    stream << ",\"source_range_stop\":" << source.range_stop;
    stream << ",\"source_reverse_start_global_frame\":"
           << witness.actual_source.reverse_start_global_frame;
    stream << ",\"source_sequence_id\":";
    write_json_string(stream, witness.actual_source.sequence_id);
    stream << ",\"source_support_height_f32_hex\":";
    write_json_string(stream, f32_hex(source.source_support_height_m));
    stream << ",\"target_generation_after\":"
           << witness.target_after.generation;
    stream << ",\"target_generation_before\":"
           << witness.target_before.generation;
    stream << ",\"target_state\":\"Held\"";
    stream << ",\"terminal_state\":\"Carry\"";
    stream << "}\n";
}

void write_headless_lifecycle_oracle(
    const std::filesystem::path& flat_database_path,
    const std::filesystem::path& full_pack_path,
    const std::filesystem::path& evidence_path) {
    using namespace interaction;
    const Database database = load_database(
        full_pack_path / "interaction_database.bin");
    const Features features = load_features(
        full_pack_path / "interaction_features.bin");
    validate_controller_interaction_pack(database, features);
    const LocomotionSnapshot flat = published_flat_snapshot(
        flat_database_path, database);
    std::ofstream evidence(evidence_path, std::ios::trunc);
    require(evidence.is_open(),
        "could not open headless lifecycle JSONL output");
    evidence << std::setprecision(std::numeric_limits<float>::max_digits10);

    constexpr std::array<std::string_view, 3> positive_cases{{
        "clear_front",
        "clear_left",
        "clear_right",
    }};
    for (std::string_view case_id : positive_cases) {
        const SmartPickupDemoScenario& scenario = scenario_named(case_id);
        const SmartPickupSlotProvenance& slot =
            selected_slot_provenance(scenario);
        for (uint32_t repeat = 0U;
             repeat < kSmartPickupDemoRepeatCount;
             ++repeat) {
            const SmartPickupLifecycleWitness witness =
                run_smart_pickup_lifecycle(
                    database, features, flat, scenario);
            write_case_result(evidence, scenario, repeat, slot, witness);
        }
    }
    evidence.flush();
    require(evidence.good(),
        "could not publish headless lifecycle JSONL output");
    std::cout << "published 30 actual-attachment case results\n";
}

}  // namespace

int main(int argc, char** argv) {
    require(argc == 1 || argc == 5,
        "usage: test_interaction_smart_pickup_scenarios "
        "[<flat-database.bin> <full-pack> --jsonl <evidence.jsonl>]");
    const std::filesystem::path flat_database_path = argc == 1
        ? std::filesystem::path("resources/database.bin")
        : std::filesystem::path(argv[1]);
    const std::filesystem::path full_pack_path = argc == 1
        ? std::filesystem::path("build/smart-pickup/full-pack")
        : std::filesystem::path(argv[2]);
    if (argc == 5) {
        require(std::string_view(argv[3]) == "--jsonl",
            "headless lifecycle oracle requires --jsonl");
    }
    test_checked_frame_conversions_and_local_row_factory();
    test_certified_registry_is_an_exact_raw_provenance_join();
    test_catalog_api_order_and_geometry_contract();
    test_materialized_scenes_and_selector_outcomes();
    test_typed_target_compatibility_policy();
    test_full_pack_lifecycle_requires_actual_attachment(
        flat_database_path, full_pack_path);
    if (argc == 5) {
        write_headless_lifecycle_oracle(
            flat_database_path, full_pack_path, argv[4]);
    }
    return 0;
}
