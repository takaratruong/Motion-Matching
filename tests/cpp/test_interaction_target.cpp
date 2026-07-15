#include "interaction_target.h"

#include <array>
#include <cmath>
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

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

#define TEST_CHECK(condition) \
    require(static_cast<bool>(condition), #condition)

bool same_position(const interaction::Transform& transform, vec3 position) {
    return transform.position.x == position.x &&
           transform.position.y == position.y &&
           transform.position.z == position.z;
}

bool exact(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool exact(quat left, quat right) {
    return left.w == right.w && left.x == right.x &&
           left.y == right.y && left.z == right.z;
}

bool exact(
    const interaction::Transform& left,
    const interaction::Transform& right) {
    return exact(left.position, right.position) &&
           exact(left.rotation, right.rotation);
}

bool exact(
    const interaction::GraspAffordance& left,
    const interaction::GraspAffordance& right) {
    return left.id == right.id && left.hand == right.hand &&
           exact(left.hand_in_object, right.hand_in_object) &&
           exact(
               left.approach_direction_object,
               right.approach_direction_object) &&
           left.clearance_radius == right.clearance_radius;
}

bool exact(
    const interaction::InteractionTarget& left,
    const interaction::InteractionTarget& right) {
    if (left.handle != right.handle ||
        !exact(left.object_world, right.object_world) ||
        left.object_profile_id != right.object_profile_id ||
        !exact(
            left.object_bounds.center_object,
            right.object_bounds.center_object) ||
        !exact(
            left.object_bounds.half_extents_object,
            right.object_bounds.half_extents_object) ||
        !exact(left.object_dimensions, right.object_dimensions) ||
        !exact(left.table_world, right.table_world) ||
        !exact(left.table_size, right.table_size) ||
        left.state != right.state ||
        left.owner_request != right.owner_request ||
        left.affordances.size() != right.affordances.size()) {
        return false;
    }
    for (size_t index = 0; index < left.affordances.size(); ++index) {
        if (!exact(left.affordances[index], right.affordances[index])) {
            return false;
        }
    }
    return true;
}

interaction::InteractionTarget make_target_with_affordance_ids(
    uint64_t id,
    uint32_t generation,
    std::initializer_list<uint32_t> affordance_ids) {
    using namespace interaction;

    InteractionTarget target{};
    target.handle = {id, generation};
    target.object_world = {vec3(0.0F, 0.75F, 0.5F), quat()};
    target.object_profile_id = 1001U;
    target.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    target.object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.04F)};
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
    target.object_bounds = {vec3(), dimensions * 0.5F};
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
    static_assert(std::is_same_v<
        decltype(InteractionTarget{}.object_profile_id), uint64_t>);
    static_assert(std::is_same_v<
        decltype(InteractionTarget{}.object_bounds), ObjectLocalBounds>);

    const PickRequest request{TargetHandle{7, 2}, 5, 42};
    TEST_CHECK(request.target == (TargetHandle{7, 2}));
    TEST_CHECK(request.affordance_id == 5);
    TEST_CHECK(request.request_id == 42);
    TEST_CHECK((TargetHandle{7, 2}) != (TargetHandle{7, 3}));

    const InteractionTarget target = make_target(7, 2);
    TEST_CHECK(target.object_profile_id == 1001U);
    TEST_CHECK(exact(target.object_bounds.center_object, vec3()));
    TEST_CHECK(exact(
        target.object_bounds.half_extents_object,
        target.object_dimensions * 0.5F));
}

void test_registry_and_resolver() {
    using namespace interaction;

    TargetRegistry registry;
    const TargetHandle handle = registry.upsert(make_target(7, 1));
    TEST_CHECK(handle.id == 7 && handle.generation == 1);
    TEST_CHECK(registry.resolve_single_target(vec3(), 1.0F) == handle);
    TEST_CHECK(registry.reserve(handle, 42));
    TEST_CHECK(registry.find(handle)->state == ObjectState::Targeted);
    TEST_CHECK(!registry.reserve(handle, 43));
    TEST_CHECK(registry.validate(handle, 42));
    registry.replace_pose(handle.id, Transform{vec3(4.0F, 0.0F, 0.0F), quat()});
    TEST_CHECK(!registry.validate(handle, 42));
    TEST_CHECK(!registry.resolve_single_target(vec3(), 1.0F).has_value());
}

void test_direct_id_lookup_ignores_generation_and_target_count() {
    using namespace interaction;

    TargetRegistry registry;
    TEST_CHECK(registry.find_by_id(80) == nullptr);

    constexpr uint32_t kSparseGeneration = 4'000'000'000U;
    const TargetHandle sparse = registry.upsert(make_target_at(
        80, kSparseGeneration, vec3(1.0F, 2.0F, 3.0F)));
    registry.upsert(make_target_at(81, 9, vec3(4.0F, 5.0F, 6.0F)));

    const TargetRegistry& read_only = registry;
    const InteractionTarget* current = read_only.find_by_id(sparse.id);
    TEST_CHECK(current != nullptr);
    TEST_CHECK(current->handle == sparse);
    TEST_CHECK(same_position(current->object_world, vec3(1.0F, 2.0F, 3.0F)));
    TEST_CHECK(read_only.find_by_id(81) != nullptr);
    TEST_CHECK(read_only.find_by_id(999) == nullptr);

    const TargetHandle replaced = registry.replace_pose(
        sparse.id, Transform{vec3(7.0F, 8.0F, 9.0F), quat()});
    current = read_only.find_by_id(sparse.id);
    TEST_CHECK(current != nullptr);
    TEST_CHECK(current->handle == replaced);
    TEST_CHECK(same_position(current->object_world, vec3(7.0F, 8.0F, 9.0F)));
}

void test_required_boundaries() {
    using namespace interaction;

    TargetRegistry registry;
    TEST_CHECK(throws_invalid_argument([&] {
        registry.upsert(make_target(0, 1));
    }));
    TEST_CHECK(throws_invalid_argument([&] {
        registry.upsert(make_target_with_dimensions(8, 1, vec3(0, 1, 1)));
    }));
    TEST_CHECK(throws_invalid_argument([&] {
        registry.upsert(make_target_with_affordance_ids(9, 1, {3, 3}));
    }));

    TargetRegistry reset_registry;
    const TargetHandle before = reset_registry.upsert(make_target(10, 4));
    const TargetHandle after = reset_registry.reset(
        before.id, Transform{vec3(1, 2, 3), quat()});
    TEST_CHECK(after == (TargetHandle{10, 5}));
    TEST_CHECK(reset_registry.find(after)->state == ObjectState::Free);
    TEST_CHECK(reset_registry.find(after)->owner_request == 0);

    TargetRegistry ambiguous;
    ambiguous.upsert(make_target(11, 1));
    ambiguous.upsert(make_target(12, 1));
    TEST_CHECK(!ambiguous.resolve_single_target(vec3(), 1.0F).has_value());
}

void test_upsert_validation_and_replacement() {
    using namespace interaction;

    TargetRegistry invalid;
    TEST_CHECK(throws_invalid_argument([&] {
        invalid.upsert(make_target(1, 0));
    }));

    InteractionTarget empty_affordances = make_target(2, 1);
    empty_affordances.affordances.clear();
    TEST_CHECK(throws_invalid_argument([&] {
        invalid.upsert(empty_affordances);
    }));
    TEST_CHECK(throws_invalid_argument([&] {
        invalid.upsert(make_target_with_dimensions(3, 1, vec3(1, -1, 1)));
    }));

    TargetRegistry registry;
    const TargetHandle original = registry.upsert(make_target_at(
        30, 7, vec3(0.0F, 0.75F, 0.25F)));
    TEST_CHECK(registry.reserve(original, 100));

    InteractionTarget replacement = make_target_at(
        30, 99, vec3(2.0F, 3.0F, 4.0F));
    replacement.state = ObjectState::Held;
    replacement.owner_request = 999;
    const TargetHandle replaced = registry.upsert(replacement);

    TEST_CHECK(replaced == (TargetHandle{30, 8}));
    TEST_CHECK(registry.find(original) == nullptr);
    TEST_CHECK(!registry.validate(original, 100));
    const InteractionTarget* stored = registry.find(replaced);
    TEST_CHECK(stored != nullptr);
    TEST_CHECK(same_position(stored->object_world, vec3(2.0F, 3.0F, 4.0F)));
    TEST_CHECK(stored->state == ObjectState::Free);
    TEST_CHECK(stored->owner_request == 0);
}

void test_registry_mutation_validation_is_strict_and_transactional() {
    using namespace interaction;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    struct InvalidTargetCase {
        void (*mutate)(InteractionTarget&);
    };
    const InvalidTargetCase invalid_targets[] = {
        {+[](InteractionTarget& target) {
            target.object_world.position.x =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.object_world.rotation = quat(0.0F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](InteractionTarget& target) {
            target.object_world.rotation = quat(2.0F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](InteractionTarget& target) {
            target.object_world.rotation.x =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.object_dimensions.y =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.object_dimensions.z = 0.0F;
        }},
        {+[](InteractionTarget& target) {
            target.object_profile_id = 0U;
        }},
        {+[](InteractionTarget& target) {
            target.object_bounds.center_object.x =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.object_bounds.center_object.z =
                std::numeric_limits<float>::infinity();
        }},
        {+[](InteractionTarget& target) {
            target.object_bounds.half_extents_object.x = 0.0F;
        }},
        {+[](InteractionTarget& target) {
            target.object_bounds.half_extents_object.y = -0.01F;
        }},
        {+[](InteractionTarget& target) {
            target.object_bounds.half_extents_object.z =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.object_bounds.half_extents_object.x =
                std::numeric_limits<float>::infinity();
        }},
        {+[](InteractionTarget& target) {
            target.table_world.position.z =
                std::numeric_limits<float>::infinity();
        }},
        {+[](InteractionTarget& target) {
            target.table_world.rotation = quat(0.0F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](InteractionTarget& target) {
            target.table_world.rotation = quat(0.5F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](InteractionTarget& target) {
            target.table_world.rotation.z =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.table_size.x =
                std::numeric_limits<float>::infinity();
        }},
        {+[](InteractionTarget& target) {
            target.table_size.y = -0.01F;
        }},
        {+[](InteractionTarget& target) {
            target.state = static_cast<ObjectState>(255U);
        }},
        {+[](InteractionTarget& target) {
            target.state = ObjectState::Free;
            target.owner_request = 1U;
        }},
        {+[](InteractionTarget& target) {
            target.state = ObjectState::Targeted;
            target.owner_request = 0U;
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().id = 0U;
        }},
        {+[](InteractionTarget& target) {
            target.affordances.push_back(target.affordances.front());
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().hand = static_cast<Hand>(2U);
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().hand_in_object.position.y =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().hand_in_object.rotation =
                quat(0.0F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().hand_in_object.rotation =
                quat(1.5F, 0.0F, 0.0F, 0.0F);
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().hand_in_object.rotation.w =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().approach_direction_object.x =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().approach_direction_object = vec3();
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().clearance_radius =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().clearance_radius = -0.001F;
        }},
    };

    for (const InvalidTargetCase& invalid : invalid_targets) {
        TargetRegistry registry;
        const TargetHandle original = registry.upsert(make_target(90, 6));
        TEST_CHECK(registry.reserve(original, 700));
        const InteractionTarget snapshot = *registry.find(original);
        InteractionTarget malformed = make_target(90, 999);
        invalid.mutate(malformed);

        TEST_CHECK(throws_invalid_argument([&] {
            (void)registry.upsert(malformed);
        }));
        const InteractionTarget* preserved = registry.find(original);
        TEST_CHECK(preserved != nullptr);
        TEST_CHECK(exact(*preserved, snapshot));
        TEST_CHECK(registry.validate(original, 700));
    }

    const std::array<Transform, 4> invalid_replacements = {{
        {vec3(nan, 1.0F, 2.0F), quat()},
        {vec3(1.0F, 2.0F, 3.0F), quat(0.0F, 0.0F, 0.0F, 0.0F)},
        {vec3(1.0F, 2.0F, 3.0F), quat(2.0F, 0.0F, 0.0F, 0.0F)},
        {vec3(1.0F, 2.0F, 3.0F), quat(1.0F, infinity, 0.0F, 0.0F)},
    }};
    for (size_t index = 0; index < invalid_replacements.size(); ++index) {
        TargetRegistry registry;
        const TargetHandle original = registry.upsert(make_target(91, 8));
        TEST_CHECK(registry.reserve(original, 701));
        const InteractionTarget snapshot = *registry.find(original);

        TEST_CHECK(throws_invalid_argument([&] {
            if (index % 2U == 0U) {
                (void)registry.replace_pose(
                    original.id, invalid_replacements[index]);
            } else {
                (void)registry.reset(
                    original.id, invalid_replacements[index]);
            }
        }));
        const InteractionTarget* preserved = registry.find(original);
        TEST_CHECK(preserved != nullptr);
        TEST_CHECK(exact(*preserved, snapshot));
        TEST_CHECK(registry.validate(original, 701));
    }
}

void test_nonfinite_mutations_reject_with_assertions_disabled() {
    using namespace interaction;
    struct NonfiniteTargetCase {
        void (*mutate)(InteractionTarget&);
    };
    const NonfiniteTargetCase invalid_targets[] = {
        {+[](InteractionTarget& target) {
            target.object_world.position.x =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.table_size.z = std::numeric_limits<float>::infinity();
        }},
        {+[](InteractionTarget& target) {
            target.object_bounds.center_object.y =
                std::numeric_limits<float>::quiet_NaN();
        }},
        {+[](InteractionTarget& target) {
            target.object_bounds.half_extents_object.x =
                std::numeric_limits<float>::infinity();
        }},
        {+[](InteractionTarget& target) {
            target.affordances.front().clearance_radius =
                std::numeric_limits<float>::quiet_NaN();
        }},
    };
    for (const NonfiniteTargetCase& invalid : invalid_targets) {
        TargetRegistry registry;
        const TargetHandle original = registry.upsert(make_target(92, 9));
        require(
            registry.reserve(original, 702),
            "nonfinite target setup reservation failed");
        const InteractionTarget snapshot = *registry.find(original);
        InteractionTarget malformed = make_target(92, 999);
        invalid.mutate(malformed);

        const bool rejected = throws_invalid_argument([&] {
            (void)registry.upsert(malformed);
        });
        require(rejected, "nonfinite target mutation was accepted");
        const InteractionTarget* preserved = registry.find(original);
        require(preserved != nullptr, "nonfinite target changed generation");
        require(
            exact(*preserved, snapshot),
            "nonfinite target changed registry contents");
        require(
            registry.validate(original, 702),
            "nonfinite target changed registry ownership");
    }

    const std::array<Transform, 2> invalid_replacements = {{
        {vec3(
             std::numeric_limits<float>::quiet_NaN(),
             1.0F,
             2.0F),
         quat()},
        {vec3(1.0F, 2.0F, 3.0F),
         quat(
             1.0F,
             std::numeric_limits<float>::infinity(),
             0.0F,
             0.0F)},
    }};
    for (Transform invalid : invalid_replacements) {
        TargetRegistry registry;
        const TargetHandle original = registry.upsert(make_target(93, 10));
        require(
            registry.reserve(original, 703),
            "nonfinite replacement setup reservation failed");
        const InteractionTarget snapshot = *registry.find(original);

        const bool rejected = throws_invalid_argument([&] {
            (void)registry.replace_pose(original.id, invalid);
        });
        require(rejected, "nonfinite replacement was accepted");
        const InteractionTarget* preserved = registry.find(original);
        require(preserved != nullptr, "nonfinite replacement changed generation");
        require(
            exact(*preserved, snapshot),
            "nonfinite replacement changed registry contents");
        require(
            registry.validate(original, 703),
            "nonfinite replacement changed registry ownership");
    }
}

void test_planar_resolution_and_free_filtering() {
    using namespace interaction;

    TargetRegistry planar;
    const TargetHandle high = planar.upsert(make_target_at(
        40, 2, vec3(0.0F, 100.0F, 0.75F)));
    TEST_CHECK(planar.resolve_single_target(vec3(0.0F, -100.0F, 0.0F), 0.75F) ==
           high);
    TEST_CHECK(!planar.resolve_single_target(vec3(), 0.74F).has_value());
    TEST_CHECK(!planar.resolve_single_target(vec3(), -1.0F).has_value());

    TargetRegistry free_only;
    const TargetHandle reserved = free_only.upsert(make_target_at(
        41, 1, vec3(0.0F, 0.0F, 0.25F)));
    const TargetHandle available = free_only.upsert(make_target_at(
        42, 1, vec3(0.0F, 0.0F, 0.50F)));
    TEST_CHECK(free_only.reserve(reserved, 200));
    TEST_CHECK(free_only.resolve_single_target(vec3(), 1.0F) == available);
}

void test_reservation_state_machine() {
    using namespace interaction;

    TargetRegistry registry;
    const TargetHandle handle = registry.upsert(make_target(50, 6));
    TEST_CHECK(!registry.reserve(handle, 0));
    TEST_CHECK(!registry.reserve(TargetHandle{50, 5}, 300));
    TEST_CHECK(registry.find(handle)->state == ObjectState::Free);

    TEST_CHECK(registry.reserve(handle, 300));
    TEST_CHECK(registry.find(handle)->handle == handle);
    TEST_CHECK(registry.find(handle)->owner_request == 300);
    TEST_CHECK(registry.validate(handle, 300));
    TEST_CHECK(!registry.validate(handle, 0));
    TEST_CHECK(!registry.validate(handle, 301));
    TEST_CHECK(!registry.attach(handle, 301));
    TEST_CHECK(!registry.hold(handle, 300));

    TEST_CHECK(registry.attach(handle, 300));
    TEST_CHECK(registry.find(handle)->state == ObjectState::Attached);
    TEST_CHECK(registry.validate(handle, 300));
    TEST_CHECK(!registry.attach(handle, 300));
    TEST_CHECK(registry.hold(handle, 300));
    TEST_CHECK(registry.find(handle)->state == ObjectState::Held);
    TEST_CHECK(!registry.hold(handle, 300));
    TEST_CHECK(!registry.release(handle, 301));

    TEST_CHECK(registry.release(handle, 300));
    TEST_CHECK(registry.find(handle)->state == ObjectState::Free);
    TEST_CHECK(registry.find(handle)->owner_request == 0);
    TEST_CHECK(!registry.validate(handle, 300));
    TEST_CHECK(!registry.release(handle, 300));

    TEST_CHECK(registry.reserve(handle, 302));
    TEST_CHECK(registry.release(handle, 302));
}

void test_pose_replacement_reset_and_affordance_lookup() {
    using namespace interaction;

    TargetRegistry registry;
    const TargetHandle original = registry.upsert(
        make_target_with_affordance_ids(60, 10, {4, 8}));
    TEST_CHECK(registry.find_affordance(original, 4)->id == 4);
    TEST_CHECK(registry.find_affordance(original, 8)->id == 8);
    TEST_CHECK(registry.find_affordance(original, 7) == nullptr);
    TEST_CHECK(registry.reserve(original, 400));

    const TargetHandle moved = registry.replace_pose(
        original.id, Transform{vec3(2.0F, 5.0F, 7.0F), quat()});
    TEST_CHECK(moved == (TargetHandle{60, 11}));
    TEST_CHECK(registry.find_affordance(original, 4) == nullptr);
    TEST_CHECK(same_position(
        registry.find(moved)->object_world, vec3(2.0F, 5.0F, 7.0F)));
    TEST_CHECK(registry.find(moved)->state == ObjectState::Free);
    TEST_CHECK(registry.find(moved)->owner_request == 0);

    TEST_CHECK(registry.reserve(moved, 401));
    const TargetHandle reset = registry.reset(
        moved.id, Transform{vec3(1.0F, 2.0F, 3.0F), quat()});
    TEST_CHECK(reset == (TargetHandle{60, 12}));
    TEST_CHECK(same_position(
        registry.find(reset)->object_world, vec3(1.0F, 2.0F, 3.0F)));
    TEST_CHECK(registry.find(reset)->state == ObjectState::Free);
    TEST_CHECK(registry.find(reset)->owner_request == 0);
}

void test_generation_overflow_is_rejected() {
    using namespace interaction;

    constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();

    TargetRegistry upsert_registry;
    const TargetHandle upsert_handle =
        upsert_registry.upsert(make_target(70, maximum));
    TEST_CHECK(upsert_registry.reserve(upsert_handle, 500));
    TEST_CHECK(throws_overflow_error([&] {
        upsert_registry.upsert(make_target(70, 1));
    }));
    TEST_CHECK(upsert_registry.validate(upsert_handle, 500));

    TargetRegistry pose_registry;
    const TargetHandle pose_handle =
        pose_registry.upsert(make_target(71, maximum));
    TEST_CHECK(throws_overflow_error([&] {
        pose_registry.replace_pose(71, Transform{vec3(), quat()});
    }));
    TEST_CHECK(throws_overflow_error([&] {
        pose_registry.reset(71, Transform{vec3(), quat()});
    }));
    TEST_CHECK(pose_registry.find(pose_handle) != nullptr);
}

}  // namespace

int main() {
    test_public_records();
    test_registry_and_resolver();
    test_direct_id_lookup_ignores_generation_and_target_count();
    test_required_boundaries();
    test_upsert_validation_and_replacement();
    test_registry_mutation_validation_is_strict_and_transactional();
    test_nonfinite_mutations_reject_with_assertions_disabled();
    test_planar_resolution_and_free_filtering();
    test_reservation_state_machine();
    test_pose_replacement_reset_and_affordance_lookup();
    test_generation_overflow_is_rejected();
}
