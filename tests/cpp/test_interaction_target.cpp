#include "interaction_target.h"

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>

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

template<class Function>
bool throws_overflow_error(Function&& function) {
    try {
        function();
    } catch (const std::overflow_error&) {
        return true;
    } catch (...) {
    }
    return false;
}

bool same_position(const interaction::Transform& transform, vec3 position) {
    return transform.position.x == position.x &&
           transform.position.y == position.y &&
           transform.position.z == position.z;
}

interaction::InteractionTarget make_target_with_affordance_ids(
    uint64_t id,
    uint32_t generation,
    std::initializer_list<uint32_t> affordance_ids) {
    using namespace interaction;

    InteractionTarget target{};
    target.handle = {id, generation};
    target.object_world = {vec3(0.0F, 0.75F, 0.5F), quat()};
    target.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    target.table_world = {vec3(0.0F, 0.70F, 0.5F), quat()};
    target.table_size = vec3(1.0F, 0.1F, 1.0F);
    for (uint32_t affordance_id : affordance_ids) {
        GraspAffordance affordance{};
        affordance.id = affordance_id;
        affordance.hand = Hand::Right;
        affordance.hand_in_object = {vec3(), quat()};
        affordance.approach_direction_object = vec3(0.0F, 0.0F, 1.0F);
        target.affordances.push_back(affordance);
    }
    return target;
}

interaction::InteractionTarget make_target(uint64_t id, uint32_t generation) {
    return make_target_with_affordance_ids(id, generation, {3});
}

interaction::InteractionTarget make_target_at(
    uint64_t id,
    uint32_t generation,
    vec3 position) {
    interaction::InteractionTarget target = make_target(id, generation);
    target.object_world.position = position;
    return target;
}

interaction::InteractionTarget make_target_with_dimensions(
    uint64_t id,
    uint32_t generation,
    vec3 dimensions) {
    interaction::InteractionTarget target = make_target(id, generation);
    target.object_dimensions = dimensions;
    return target;
}

void test_public_records() {
    using namespace interaction;

    static_assert(std::is_same_v<std::underlying_type_t<Hand>, uint8_t>);
    static_assert(static_cast<uint8_t>(Hand::Left) == 0U);
    static_assert(static_cast<uint8_t>(Hand::Right) == 1U);
    static_assert(std::is_same_v<decltype(PickRequest{}.target), TargetHandle>);
    static_assert(std::is_same_v<decltype(PickRequest{}.affordance_id), uint32_t>);
    static_assert(std::is_same_v<decltype(PickRequest{}.request_id), uint64_t>);

    const PickRequest request{TargetHandle{7, 2}, 5, 42};
    assert(request.target == (TargetHandle{7, 2}));
    assert(request.affordance_id == 5);
    assert(request.request_id == 42);
    assert((TargetHandle{7, 2}) != (TargetHandle{7, 3}));
}

void test_registry_and_resolver() {
    using namespace interaction;

    TargetRegistry registry;
    const TargetHandle handle = registry.upsert(make_target(7, 1));
    assert(handle.id == 7 && handle.generation == 1);
    assert(registry.resolve_single_target(vec3(), 1.0F) == handle);
    assert(registry.reserve(handle, 42));
    assert(registry.find(handle)->state == ObjectState::Targeted);
    assert(!registry.reserve(handle, 43));
    assert(registry.validate(handle, 42));
    registry.replace_pose(handle.id, Transform{vec3(4.0F, 0.0F, 0.0F), quat()});
    assert(!registry.validate(handle, 42));
    assert(!registry.resolve_single_target(vec3(), 1.0F).has_value());
}

void test_direct_id_lookup_ignores_generation_and_target_count() {
    using namespace interaction;

    TargetRegistry registry;
    assert(registry.find_by_id(80) == nullptr);

    constexpr uint32_t kSparseGeneration = 4'000'000'000U;
    const TargetHandle sparse = registry.upsert(make_target_at(
        80, kSparseGeneration, vec3(1.0F, 2.0F, 3.0F)));
    registry.upsert(make_target_at(81, 9, vec3(4.0F, 5.0F, 6.0F)));

    const TargetRegistry& read_only = registry;
    const InteractionTarget* current = read_only.find_by_id(sparse.id);
    assert(current != nullptr);
    assert(current->handle == sparse);
    assert(same_position(current->object_world, vec3(1.0F, 2.0F, 3.0F)));
    assert(read_only.find_by_id(81) != nullptr);
    assert(read_only.find_by_id(999) == nullptr);

    const TargetHandle replaced = registry.replace_pose(
        sparse.id, Transform{vec3(7.0F, 8.0F, 9.0F), quat()});
    current = read_only.find_by_id(sparse.id);
    assert(current != nullptr);
    assert(current->handle == replaced);
    assert(same_position(current->object_world, vec3(7.0F, 8.0F, 9.0F)));
}

void test_required_boundaries() {
    using namespace interaction;

    TargetRegistry registry;
    assert(throws_invalid_argument([&] {
        registry.upsert(make_target(0, 1));
    }));
    assert(throws_invalid_argument([&] {
        registry.upsert(make_target_with_dimensions(8, 1, vec3(0, 1, 1)));
    }));
    assert(throws_invalid_argument([&] {
        registry.upsert(make_target_with_affordance_ids(9, 1, {3, 3}));
    }));

    TargetRegistry reset_registry;
    const TargetHandle before = reset_registry.upsert(make_target(10, 4));
    const TargetHandle after = reset_registry.reset(
        before.id, Transform{vec3(1, 2, 3), quat()});
    assert(after == (TargetHandle{10, 5}));
    assert(reset_registry.find(after)->state == ObjectState::Free);
    assert(reset_registry.find(after)->owner_request == 0);

    TargetRegistry ambiguous;
    ambiguous.upsert(make_target(11, 1));
    ambiguous.upsert(make_target(12, 1));
    assert(!ambiguous.resolve_single_target(vec3(), 1.0F).has_value());
}

void test_upsert_validation_and_replacement() {
    using namespace interaction;

    TargetRegistry invalid;
    assert(throws_invalid_argument([&] {
        invalid.upsert(make_target(1, 0));
    }));

    InteractionTarget empty_affordances = make_target(2, 1);
    empty_affordances.affordances.clear();
    assert(throws_invalid_argument([&] {
        invalid.upsert(empty_affordances);
    }));
    assert(throws_invalid_argument([&] {
        invalid.upsert(make_target_with_dimensions(3, 1, vec3(1, -1, 1)));
    }));

    TargetRegistry registry;
    const TargetHandle original = registry.upsert(make_target_at(
        30, 7, vec3(0.0F, 0.75F, 0.25F)));
    assert(registry.reserve(original, 100));

    InteractionTarget replacement = make_target_at(
        30, 99, vec3(2.0F, 3.0F, 4.0F));
    replacement.state = ObjectState::Held;
    replacement.owner_request = 999;
    const TargetHandle replaced = registry.upsert(replacement);

    assert(replaced == (TargetHandle{30, 8}));
    assert(registry.find(original) == nullptr);
    assert(!registry.validate(original, 100));
    const InteractionTarget* stored = registry.find(replaced);
    assert(stored != nullptr);
    assert(same_position(stored->object_world, vec3(2.0F, 3.0F, 4.0F)));
    assert(stored->state == ObjectState::Free);
    assert(stored->owner_request == 0);
}

void test_planar_resolution_and_free_filtering() {
    using namespace interaction;

    TargetRegistry planar;
    const TargetHandle high = planar.upsert(make_target_at(
        40, 2, vec3(0.0F, 100.0F, 0.75F)));
    assert(planar.resolve_single_target(vec3(0.0F, -100.0F, 0.0F), 0.75F) ==
           high);
    assert(!planar.resolve_single_target(vec3(), 0.74F).has_value());
    assert(!planar.resolve_single_target(vec3(), -1.0F).has_value());

    TargetRegistry free_only;
    const TargetHandle reserved = free_only.upsert(make_target_at(
        41, 1, vec3(0.0F, 0.0F, 0.25F)));
    const TargetHandle available = free_only.upsert(make_target_at(
        42, 1, vec3(0.0F, 0.0F, 0.50F)));
    assert(free_only.reserve(reserved, 200));
    assert(free_only.resolve_single_target(vec3(), 1.0F) == available);
}

void test_reservation_state_machine() {
    using namespace interaction;

    TargetRegistry registry;
    const TargetHandle handle = registry.upsert(make_target(50, 6));
    assert(!registry.reserve(handle, 0));
    assert(!registry.reserve(TargetHandle{50, 5}, 300));
    assert(registry.find(handle)->state == ObjectState::Free);

    assert(registry.reserve(handle, 300));
    assert(registry.find(handle)->handle == handle);
    assert(registry.find(handle)->owner_request == 300);
    assert(registry.validate(handle, 300));
    assert(!registry.validate(handle, 0));
    assert(!registry.validate(handle, 301));
    assert(!registry.attach(handle, 301));
    assert(!registry.hold(handle, 300));

    assert(registry.attach(handle, 300));
    assert(registry.find(handle)->state == ObjectState::Attached);
    assert(registry.validate(handle, 300));
    assert(!registry.attach(handle, 300));
    assert(registry.hold(handle, 300));
    assert(registry.find(handle)->state == ObjectState::Held);
    assert(!registry.hold(handle, 300));
    assert(!registry.release(handle, 301));

    assert(registry.release(handle, 300));
    assert(registry.find(handle)->state == ObjectState::Free);
    assert(registry.find(handle)->owner_request == 0);
    assert(!registry.validate(handle, 300));
    assert(!registry.release(handle, 300));

    assert(registry.reserve(handle, 302));
    assert(registry.release(handle, 302));
}

void test_pose_replacement_reset_and_affordance_lookup() {
    using namespace interaction;

    TargetRegistry registry;
    const TargetHandle original = registry.upsert(
        make_target_with_affordance_ids(60, 10, {4, 8}));
    assert(registry.find_affordance(original, 4)->id == 4);
    assert(registry.find_affordance(original, 8)->id == 8);
    assert(registry.find_affordance(original, 7) == nullptr);
    assert(registry.reserve(original, 400));

    const TargetHandle moved = registry.replace_pose(
        original.id, Transform{vec3(2.0F, 5.0F, 7.0F), quat()});
    assert(moved == (TargetHandle{60, 11}));
    assert(registry.find_affordance(original, 4) == nullptr);
    assert(same_position(
        registry.find(moved)->object_world, vec3(2.0F, 5.0F, 7.0F)));
    assert(registry.find(moved)->state == ObjectState::Free);
    assert(registry.find(moved)->owner_request == 0);

    assert(registry.reserve(moved, 401));
    const TargetHandle reset = registry.reset(
        moved.id, Transform{vec3(1.0F, 2.0F, 3.0F), quat()});
    assert(reset == (TargetHandle{60, 12}));
    assert(same_position(
        registry.find(reset)->object_world, vec3(1.0F, 2.0F, 3.0F)));
    assert(registry.find(reset)->state == ObjectState::Free);
    assert(registry.find(reset)->owner_request == 0);
}

void test_generation_overflow_is_rejected() {
    using namespace interaction;

    constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();

    TargetRegistry upsert_registry;
    const TargetHandle upsert_handle =
        upsert_registry.upsert(make_target(70, maximum));
    assert(upsert_registry.reserve(upsert_handle, 500));
    assert(throws_overflow_error([&] {
        upsert_registry.upsert(make_target(70, 1));
    }));
    assert(upsert_registry.validate(upsert_handle, 500));

    TargetRegistry pose_registry;
    const TargetHandle pose_handle =
        pose_registry.upsert(make_target(71, maximum));
    assert(throws_overflow_error([&] {
        pose_registry.replace_pose(71, Transform{vec3(), quat()});
    }));
    assert(throws_overflow_error([&] {
        pose_registry.reset(71, Transform{vec3(), quat()});
    }));
    assert(pose_registry.find(pose_handle) != nullptr);
}

}  // namespace

int main() {
    test_public_records();
    test_registry_and_resolver();
    test_direct_id_lookup_ignores_generation_and_target_count();
    test_required_boundaries();
    test_upsert_validation_and_replacement();
    test_planar_resolution_and_free_filtering();
    test_reservation_state_machine();
    test_pose_replacement_reset_and_affordance_lookup();
    test_generation_overflow_is_rejected();
}
