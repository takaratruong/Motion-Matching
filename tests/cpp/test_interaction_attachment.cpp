#include "interaction_attachment.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

constexpr float kTolerance = 1.0e-5F;

bool near(float left, float right, float tolerance = kTolerance) {
    return std::abs(left - right) <= tolerance;
}

bool near(vec3 left, vec3 right, float tolerance = kTolerance) {
    return near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near(quat left, quat right, float tolerance = kTolerance) {
    return quat_angle_between(left, right) <= tolerance;
}

bool near(
    const interaction::Transform& left,
    const interaction::Transform& right,
    float tolerance = kTolerance) {
    return near(left.position, right.position, tolerance) &&
           near(left.rotation, right.rotation, tolerance);
}

bool exact(
    const interaction::Transform& left,
    const interaction::Transform& right) {
    return left.position.x == right.position.x &&
           left.position.y == right.position.y &&
           left.position.z == right.position.z &&
           left.rotation.w == right.rotation.w &&
           left.rotation.x == right.rotation.x &&
           left.rotation.y == right.rotation.y &&
           left.rotation.z == right.rotation.z;
}

bool exact(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
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

struct AttachmentFixture {
    interaction::TargetRegistry registry;
    interaction::InteractionTarget target;
    interaction::PickRequest request;
    interaction::GraspAffordance affordance;
};

AttachmentFixture make_fixture(uint32_t generation = 4U) {
    using namespace interaction;

    AttachmentFixture fixture{};
    InteractionTarget target{};
    target.handle = {31, generation};
    target.object_world = {
        vec3(0.35F, 0.72F, -0.45F),
        quat_from_angle_axis(0.31F, vec3(0.0F, 1.0F, 0.0F)),
    };
    target.object_profile_id = 2001U;
    target.object_dimensions = vec3(0.08F, 0.20F, 0.12F);
    target.object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.06F)};
    target.table_world = {
        vec3(0.0F, 0.65F, -0.30F),
        quat_from_angle_axis(-0.13F, vec3(0.0F, 1.0F, 0.0F)),
    };
    target.table_size = vec3(1.2F, 0.10F, 0.8F);
    GraspAffordance affordance{};
    affordance.id = 9;
    affordance.hand = Hand::Right;
    affordance.hand_in_object = {
        vec3(0.025F, 0.085F, -0.015F),
        quat_from_angle_axis(-0.22F, vec3(1.0F, 0.0F, 0.0F)),
    };
    affordance.approach_direction_object = vec3(0.0F, 0.0F, 1.0F);
    affordance.clearance_radius = 0.04F;
    target.affordances = {affordance};

    const TargetHandle handle = fixture.registry.upsert(target);
    assert(target.object_profile_id != 0U);
    assert(near(target.object_bounds.center_object, vec3()));
    assert(near(
        target.object_bounds.half_extents_object,
        target.object_dimensions * 0.5F));
    fixture.request = {handle, affordance.id, 7001};
    assert(fixture.registry.reserve(handle, fixture.request.request_id));
    fixture.target = *fixture.registry.find(handle);
    fixture.affordance = *fixture.registry.find_affordance(
        handle, affordance.id);
    return fixture;
}

interaction::ContactMeasurement valid_measurement(
    const AttachmentFixture& fixture) {
    using namespace interaction;
    ContactMeasurement measurement{};
    measurement.target = fixture.request.target;
    measurement.hand = fixture.affordance.hand;
    measurement.hand_world = compose(
        fixture.target.object_world, fixture.affordance.hand_in_object);
    measurement.position_error_m = 0.0F;
    measurement.orientation_error_radians = 0.0F;
    measurement.stable_contact_event = true;
    measurement.hand_contact = true;
    measurement.joints_valid = true;
    measurement.clearance_valid = true;
    return measurement;
}

interaction::ContactMeasurement measurement_for_object(
    const AttachmentFixture& fixture,
    const interaction::Transform& object_world) {
    interaction::ContactMeasurement measurement = valid_measurement(fixture);
    measurement.hand_world = interaction::compose(
        object_world, fixture.affordance.hand_in_object);
    return measurement;
}

interaction::Transform drive_to_held(
    AttachmentFixture& fixture,
    interaction::AttachmentController& attachment) {
    using namespace interaction;

    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    assert(attachment.try_contact(valid_measurement(fixture)));
    Transform held = fixture.target.object_world;
    held.position = held.position + vec3(0.20F, 0.20F, -0.10F);
    held.rotation = quat_from_angle_axis(
        -0.42F, vec3(0.0F, 1.0F, 0.0F));
    attachment.update(measurement_for_object(fixture, held), 1.0F);
    assert(attachment.state() == ObjectState::Held);
    assert(attachment.result() == ResultCode::Succeeded);
    assert(attachment.reason() == Reason::None);
    return held;
}

struct AttachmentSnapshot {
    interaction::Transform object_world{};
    interaction::ObjectState state = interaction::ObjectState::Free;
    interaction::ResultCode result = interaction::ResultCode::None;
    interaction::Reason reason = interaction::Reason::None;
    float held_seconds = 0.0F;
    interaction::InteractionTarget registry_target{};
};

AttachmentSnapshot snapshot_attachment(
    const AttachmentFixture& fixture,
    const interaction::AttachmentController& attachment) {
    const interaction::InteractionTarget* stored =
        fixture.registry.find_by_id(fixture.request.target.id);
    assert(stored != nullptr);
    return {
        attachment.object_world(),
        attachment.state(),
        attachment.result(),
        attachment.reason(),
        attachment.held_seconds(),
        *stored,
    };
}

void assert_attachment_unchanged(
    const AttachmentFixture& fixture,
    const interaction::AttachmentController& attachment,
    const AttachmentSnapshot& snapshot) {
    assert(exact(attachment.object_world(), snapshot.object_world));
    assert(attachment.state() == snapshot.state);
    assert(attachment.result() == snapshot.result);
    assert(attachment.reason() == snapshot.reason);
    assert(attachment.held_seconds() == snapshot.held_seconds);
    const interaction::InteractionTarget* stored =
        fixture.registry.find_by_id(fixture.request.target.id);
    assert(stored != nullptr);
    assert(exact(*stored, snapshot.registry_target));
}

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

template<class Mutate>
void expect_contact_gate_failure(
    Mutate&& mutate,
    interaction::Reason expected_reason) {
    using namespace interaction;
    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry, AttachmentConfig{});
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    const Transform original = attachment.object_world();
    ContactMeasurement measurement = valid_measurement(fixture);
    measurement.hand_world = {
        vec3(8.0F, 9.0F, -7.0F),
        quat_from_angle_axis(1.2F, vec3(0.0F, 0.0F, 1.0F)),
    };
    mutate(fixture, measurement);

    assert(!attachment.try_contact(measurement));
    assert(near(attachment.object_world(), original));
    assert(attachment.state() != ObjectState::Attached);
    assert(attachment.state() != ObjectState::Held);
    assert(attachment.result() == ResultCode::Failed);
    assert(attachment.reason() == expected_reason);
    assert(near(attachment.held_seconds(), 0.0F));
}

void test_frozen_public_contract_and_defaults() {
    using namespace interaction;

    static_assert(std::is_same_v<std::underlying_type_t<ResultCode>, uint8_t>);
    static_assert(static_cast<uint8_t>(ResultCode::None) == 0U);
    static_assert(static_cast<uint8_t>(ResultCode::Accepted) == 1U);
    static_assert(static_cast<uint8_t>(ResultCode::Succeeded) == 2U);
    static_assert(static_cast<uint8_t>(ResultCode::Rejected) == 3U);
    static_assert(static_cast<uint8_t>(ResultCode::Cancelled) == 4U);
    static_assert(static_cast<uint8_t>(ResultCode::Failed) == 5U);
    static_assert(static_cast<uint8_t>(ResultCode::Reset) == 6U);
    static_assert(std::is_same_v<
        decltype(ContactMeasurement{}.target), TargetHandle>);
    static_assert(std::is_same_v<decltype(ContactMeasurement{}.hand), Hand>);
    static_assert(std::is_same_v<
        decltype(ContactMeasurement{}.hand_world), Transform>);
    static_assert(std::is_same_v<
        decltype(ContactMeasurement{}.position_error_m), float>);
    static_assert(std::is_same_v<
        decltype(ContactMeasurement{}.orientation_error_radians), float>);
    static_assert(std::is_same_v<
        decltype(ContactMeasurement{}.stable_contact_event), bool>);
    static_assert(std::is_same_v<
        decltype(ContactMeasurement{}.hand_contact), bool>);
    static_assert(std::is_same_v<
        decltype(ContactMeasurement{}.joints_valid), bool>);
    static_assert(std::is_same_v<
        decltype(ContactMeasurement{}.clearance_valid), bool>);
    static_assert(std::is_constructible_v<
        AttachmentController, TargetRegistry&>);
    static_assert(std::is_constructible_v<
        AttachmentController, TargetRegistry&, AttachmentConfig>);
    static_assert(std::is_same_v<
        decltype(&AttachmentController::begin),
        bool (AttachmentController::*)(
            const InteractionTarget&,
            const PickRequest&,
            const GraspAffordance&,
            float)>);
    static_assert(std::is_same_v<
        decltype(&AttachmentController::try_contact),
        bool (AttachmentController::*)(const ContactMeasurement&)>);
    static_assert(std::is_same_v<
        decltype(&AttachmentController::update),
        void (AttachmentController::*)(const ContactMeasurement&, float)>);
    static_assert(std::is_same_v<
        decltype(&AttachmentController::reset),
        TargetHandle (AttachmentController::*)(Transform)>);
    static_assert(std::is_same_v<
        decltype(&AttachmentController::commit_place),
        std::optional<TargetHandle> (AttachmentController::*)(
            Transform, PlacedSupportContext)>);
    static_assert(std::is_same_v<
        decltype(&AttachmentController::object_world),
        Transform (AttachmentController::*)() const>);
    static_assert(std::is_same_v<
        decltype(&AttachmentController::state),
        ObjectState (AttachmentController::*)() const>);
    static_assert(std::is_same_v<
        decltype(&AttachmentController::result),
        ResultCode (AttachmentController::*)() const>);
    static_assert(std::is_same_v<
        decltype(&AttachmentController::reason),
        Reason (AttachmentController::*)() const>);
    static_assert(std::is_same_v<
        decltype(&AttachmentController::held_seconds),
        float (AttachmentController::*)() const>);

    const AttachmentConfig config{};
    assert(config.maximum_position_error_m == 0.04F);
    assert(config.maximum_orientation_error_radians == 0.261799388F);
    assert(config.required_lift_m == 0.15F);
    assert(config.required_hold_seconds == 1.00F);

    TargetRegistry registry;
    const AttachmentController unstarted(registry);
    assert(unstarted.state() == ObjectState::Free);
    assert(unstarted.result() == ResultCode::None);
    assert(unstarted.reason() == Reason::None);
    assert(near(unstarted.held_seconds(), 0.0F));
    assert(near(unstarted.object_world(), Transform{}));
}

void test_every_contact_gate_rejects_without_teleporting() {
    using namespace interaction;

    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.stable_contact_event = false;
        },
        Reason::LostContact);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            ++measurement.target.generation;
        },
        Reason::TargetChanged);
    expect_contact_gate_failure(
        [](AttachmentFixture& fixture, ContactMeasurement&) {
            assert(fixture.registry.release(
                fixture.request.target, fixture.request.request_id));
        },
        Reason::TargetChanged);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.hand = Hand::Left;
        },
        Reason::LostContact);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.hand_contact = false;
        },
        Reason::LostContact);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.position_error_m = 0.040001F;
        },
        Reason::ContactPosition);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.orientation_error_radians = 0.261800F;
        },
        Reason::ContactOrientation);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.joints_valid = false;
        },
        Reason::JointLimit);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.clearance_valid = false;
        },
        Reason::BlockedPath);
}

void test_valid_contact_attaches_at_inclusive_boundaries() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    assert(attachment.state() == ObjectState::Targeted);
    assert(attachment.result() == ResultCode::Accepted);
    assert(attachment.reason() == Reason::None);
    assert(near(attachment.object_world(), fixture.target.object_world));

    ContactMeasurement contact = valid_measurement(fixture);
    contact.position_error_m = 0.04F;
    contact.orientation_error_radians = 0.261799388F;
    assert(attachment.try_contact(contact));

    assert(attachment.state() == ObjectState::Attached);
    assert(fixture.registry.find(fixture.request.target)->state ==
           ObjectState::Attached);
    assert(attachment.result() == ResultCode::Accepted);
    assert(attachment.reason() == Reason::None);
    assert(near(attachment.object_world(), fixture.target.object_world));
    assert(near(
        attachment.object_world(),
        compose(contact.hand_world, inverse(fixture.affordance.hand_in_object))));
    assert(near(attachment.held_seconds(), 0.0F));
}

void test_failed_contact_can_be_remeasured_without_teleporting() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    const Transform original = attachment.object_world();
    ContactMeasurement rejected = valid_measurement(fixture);
    rejected.position_error_m = 0.041F;
    rejected.hand_world.position =
        rejected.hand_world.position + vec3(2.0F, 3.0F, -4.0F);

    assert(!attachment.try_contact(rejected));
    assert(attachment.reason() == Reason::ContactPosition);
    assert(attachment.result() == ResultCode::Failed);
    assert(attachment.state() == ObjectState::Targeted);
    assert(near(attachment.object_world(), original));
    assert(fixture.registry.validate(
        fixture.request.target, fixture.request.request_id));

    assert(attachment.try_contact(valid_measurement(fixture)));
    assert(attachment.state() == ObjectState::Attached);
    assert(attachment.result() == ResultCode::Accepted);
    assert(attachment.reason() == Reason::None);
    assert(near(attachment.object_world(), original));
}

void test_object_follows_grasp_and_hold_requires_continuous_lift() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    const float pre_lift_height = fixture.target.object_world.position.y;
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        pre_lift_height));
    assert(attachment.try_contact(valid_measurement(fixture)));

    Transform lifted = fixture.target.object_world;
    lifted.position = vec3(-0.20F, pre_lift_height + 0.15F, 0.55F);
    lifted.rotation = quat_from_angle_axis(
        -0.47F, vec3(0.0F, 1.0F, 0.0F));
    ContactMeasurement lifted_measurement = measurement_for_object(
        fixture, lifted);
    attachment.update(lifted_measurement, 0.25F);
    assert(attachment.state() == ObjectState::Attached);
    assert(near(attachment.object_world(), lifted));
    assert(near(attachment.held_seconds(), 0.25F));

    Transform below = lifted;
    below.position.y = pre_lift_height + 0.149F;
    attachment.update(measurement_for_object(fixture, below), 0.50F);
    assert(attachment.state() == ObjectState::Attached);
    assert(near(attachment.object_world(), below));
    assert(near(attachment.held_seconds(), 0.0F));

    Transform raised = lifted;
    raised.position = vec3(0.65F, pre_lift_height + 0.18F, -0.10F);
    raised.rotation = quat_normalize(quat_mul(
        quat_from_angle_axis(0.36F, vec3(1.0F, 0.0F, 0.0F)),
        quat_from_angle_axis(-0.29F, vec3(0.0F, 1.0F, 0.0F))));
    ContactMeasurement raised_measurement = measurement_for_object(
        fixture, raised);
    for (int update = 0; update < 3; ++update) {
        attachment.update(raised_measurement, 0.25F);
        assert(attachment.state() == ObjectState::Attached);
        assert(near(
            attachment.held_seconds(),
            0.25F * static_cast<float>(update + 1)));
    }
    attachment.update(raised_measurement, 0.25F);

    assert(attachment.state() == ObjectState::Held);
    assert(fixture.registry.find(fixture.request.target)->state ==
           ObjectState::Held);
    assert(attachment.result() == ResultCode::Succeeded);
    assert(attachment.reason() == Reason::None);
    assert(near(attachment.held_seconds(), 1.0F));
    assert(near(attachment.object_world(), raised));
    assert(near(
        attachment.object_world(),
        compose(
            raised_measurement.hand_world,
            inverse(fixture.affordance.hand_in_object))));

    Transform carried = raised;
    carried.position.x += 0.25F;
    carried.position.y += 0.05F;
    ContactMeasurement carried_measurement = measurement_for_object(
        fixture, carried);
    attachment.update(carried_measurement, 0.40F);
    assert(attachment.state() == ObjectState::Held);
    assert(attachment.result() == ResultCode::Succeeded);
    assert(attachment.held_seconds() >= 1.0F);
    assert(near(attachment.object_world(), carried));
}

void test_post_attach_updates_require_contact_but_not_a_second_event() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    assert(attachment.try_contact(valid_measurement(fixture)));

    Transform moved = fixture.target.object_world;
    moved.position = moved.position + vec3(0.20F, 0.16F, -0.15F);
    ContactMeasurement continuing = measurement_for_object(fixture, moved);
    continuing.stable_contact_event = false;
    attachment.update(continuing, 0.30F);
    assert(attachment.state() == ObjectState::Attached);
    assert(attachment.reason() == Reason::None);
    assert(near(attachment.object_world(), moved));
    assert(near(attachment.held_seconds(), 0.30F));

    const Transform last_valid = attachment.object_world();
    ContactMeasurement lost = continuing;
    lost.hand_contact = false;
    lost.hand_world.position =
        lost.hand_world.position + vec3(4.0F, 4.0F, 4.0F);
    attachment.update(lost, 0.70F);
    assert(attachment.state() == ObjectState::Attached);
    assert(attachment.result() == ResultCode::Failed);
    assert(attachment.reason() == Reason::LostContact);
    assert(near(attachment.object_world(), last_valid));
    assert(near(attachment.held_seconds(), 0.0F));
    const InteractionTarget* registry_after_failure =
        fixture.registry.find(fixture.request.target);
    assert(registry_after_failure != nullptr);
    const InteractionTarget after_failure = *registry_after_failure;
    assert(after_failure.state == ObjectState::Attached);
    assert(after_failure.owner_request == fixture.request.request_id);

    Transform later_object = moved;
    later_object.position =
        later_object.position + vec3(-0.30F, 0.10F, 0.25F);
    ContactMeasurement later_valid = measurement_for_object(
        fixture, later_object);
    later_valid.stable_contact_event = false;
    attachment.update(later_valid, 0.70F);

    assert(attachment.state() == ObjectState::Attached);
    assert(attachment.result() == ResultCode::Failed);
    assert(attachment.reason() == Reason::LostContact);
    assert(near(attachment.object_world(), last_valid));
    assert(near(attachment.held_seconds(), 0.0F));
    const InteractionTarget* after_retry =
        fixture.registry.find(fixture.request.target);
    assert(after_retry != nullptr);
    assert(after_retry->handle == after_failure.handle);
    assert(after_retry->state == after_failure.state);
    assert(after_retry->owner_request == after_failure.owner_request);
    assert(exact(after_retry->object_world, after_failure.object_world));
}

void test_held_contact_loss_is_terminal_and_freezes_last_valid_state() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    assert(attachment.try_contact(valid_measurement(fixture)));
    Transform held_object = fixture.target.object_world;
    held_object.position =
        held_object.position + vec3(0.25F, 0.20F, -0.15F);
    held_object.rotation = quat_from_angle_axis(
        -0.42F, vec3(0.0F, 1.0F, 0.0F));
    ContactMeasurement held_measurement = measurement_for_object(
        fixture, held_object);
    attachment.update(held_measurement, 1.0F);
    assert(attachment.state() == ObjectState::Held);
    assert(attachment.result() == ResultCode::Succeeded);
    const Transform last_valid = attachment.object_world();
    const float held_seconds = attachment.held_seconds();
    const InteractionTarget* registry_before_failure =
        fixture.registry.find(fixture.request.target);
    assert(registry_before_failure != nullptr);
    const InteractionTarget registry_held = *registry_before_failure;
    assert(registry_held.state == ObjectState::Held);

    ContactMeasurement lost = held_measurement;
    lost.hand_contact = false;
    lost.hand_world.position =
        lost.hand_world.position + vec3(5.0F, -3.0F, 4.0F);
    attachment.update(lost, 0.25F);

    assert(attachment.state() == ObjectState::Held);
    assert(attachment.result() == ResultCode::Failed);
    assert(attachment.reason() == Reason::LostContact);
    assert(near(attachment.object_world(), last_valid));
    assert(near(attachment.held_seconds(), held_seconds));
    const InteractionTarget* after_failure =
        fixture.registry.find(fixture.request.target);
    assert(after_failure != nullptr);
    assert(after_failure->handle == registry_held.handle);
    assert(after_failure->state == ObjectState::Held);
    assert(after_failure->owner_request == registry_held.owner_request);
    assert(exact(after_failure->object_world, registry_held.object_world));

    Transform later_object = held_object;
    later_object.position =
        later_object.position + vec3(-0.60F, 0.30F, 0.45F);
    ContactMeasurement later_valid = measurement_for_object(
        fixture, later_object);
    later_valid.stable_contact_event = false;
    attachment.update(later_valid, 0.50F);

    assert(attachment.state() == ObjectState::Held);
    assert(attachment.result() == ResultCode::Failed);
    assert(attachment.reason() == Reason::LostContact);
    assert(near(attachment.object_world(), last_valid));
    assert(near(attachment.held_seconds(), held_seconds));
    const InteractionTarget* after_retry =
        fixture.registry.find(fixture.request.target);
    assert(after_retry != nullptr);
    assert(after_retry->handle == registry_held.handle);
    assert(after_retry->state == ObjectState::Held);
    assert(after_retry->owner_request == registry_held.owner_request);
    assert(exact(after_retry->object_world, registry_held.object_world));
}

void test_generation_change_fails_without_attaching_replacement() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    assert(attachment.try_contact(valid_measurement(fixture)));
    const Transform last_valid = attachment.object_world();
    const Transform replacement_world{
        vec3(-1.0F, 1.25F, 2.5F),
        quat_from_angle_axis(0.73F, vec3(0.0F, 1.0F, 0.0F)),
    };
    const TargetHandle replacement = fixture.registry.replace_pose(
        fixture.request.target.id, replacement_world);
    assert(replacement.id == fixture.request.target.id);
    assert(replacement.generation == fixture.request.target.generation + 1U);

    ContactMeasurement stale = valid_measurement(fixture);
    stale.hand_world.position =
        stale.hand_world.position + vec3(3.0F, 3.0F, 3.0F);
    attachment.update(stale, 0.50F);

    assert(attachment.result() == ResultCode::Failed);
    assert(attachment.reason() == Reason::TargetChanged);
    assert(near(attachment.object_world(), last_valid));
    assert(fixture.registry.find(fixture.request.target) == nullptr);
    const InteractionTarget* replacement_target =
        fixture.registry.find(replacement);
    assert(replacement_target != nullptr);
    assert(replacement_target->state == ObjectState::Free);
    assert(replacement_target->owner_request == 0);
    assert(exact(replacement_target->object_world, replacement_world));
}

void test_reset_restores_exact_pose_and_increments_generation() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    assert(attachment.try_contact(valid_measurement(fixture)));
    Transform lifted = fixture.target.object_world;
    lifted.position.y += 0.20F;
    attachment.update(measurement_for_object(fixture, lifted), 0.40F);
    assert(near(attachment.held_seconds(), 0.40F));

    const Transform restored{
        vec3(-0.375F, 0.8125F, 1.0625F),
        quat_from_angle_axis(-0.625F, vec3(0.0F, 1.0F, 0.0F)),
    };
    const TargetHandle reset = attachment.reset(restored);

    assert(reset.id == fixture.request.target.id);
    assert(reset.generation == fixture.request.target.generation + 1U);
    assert(fixture.registry.find(fixture.request.target) == nullptr);
    const InteractionTarget* target = fixture.registry.find(reset);
    assert(target != nullptr);
    assert(target->state == ObjectState::Free);
    assert(target->owner_request == 0);
    assert(exact(target->object_world, restored));
    assert(exact(attachment.object_world(), restored));
    assert(attachment.state() == ObjectState::Free);
    assert(attachment.result() == ResultCode::Reset);
    assert(attachment.reason() == Reason::Reset);
    assert(near(attachment.held_seconds(), 0.0F));
    assert(!fixture.registry.validate(
        fixture.request.target, fixture.request.request_id));
}

void test_commit_place_atomically_releases_to_destination_support() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    (void)drive_to_held(fixture, attachment);
    const TargetHandle old_handle = fixture.request.target;
    const InteractionTarget before = *fixture.registry.find(old_handle);
    const Transform placed{
        vec3(1.0F, 0.82F, 4.0F),
        quat_from_angle_axis(0.37F, vec3(0.0F, 1.0F, 0.0F)),
    };
    const PlacedSupportContext destination{
        Transform{
            vec3(0.0F, 0.70F, 4.0F),
            quat_from_angle_axis(-0.21F, vec3(0.0F, 1.0F, 0.0F)),
        },
        vec3(2.0F, 0.04F, 0.60F),
    };

    const auto next = attachment.commit_place(placed, destination);

    assert(next.has_value());
    assert(next->id == old_handle.id);
    assert(next->generation == old_handle.generation + 1U);
    assert(attachment.state() == ObjectState::Free);
    assert(attachment.result() == ResultCode::Succeeded);
    assert(attachment.reason() == Reason::None);
    assert(exact(attachment.object_world(), placed));
    assert(attachment.held_seconds() == 0.0F);
    assert(fixture.registry.find(old_handle) == nullptr);
    const InteractionTarget* stored = fixture.registry.find(*next);
    assert(stored != nullptr);
    assert(stored->state == ObjectState::Free);
    assert(stored->owner_request == 0U);
    assert(exact(stored->object_world, placed));
    assert(exact(stored->table_world, destination.table_world));
    assert(exact(stored->table_size, destination.table_size));
    InteractionTarget expected = before;
    expected.handle = *next;
    expected.object_world = placed;
    expected.table_world = destination.table_world;
    expected.table_size = destination.table_size;
    expected.state = ObjectState::Free;
    expected.owner_request = 0U;
    assert(exact(*stored, expected));

    const AttachmentSnapshot committed = snapshot_attachment(
        fixture, attachment);
    assert(!attachment.commit_place(placed, destination).has_value());
    assert_attachment_unchanged(fixture, attachment, committed);
}

void test_commit_place_registry_rejections_preserve_local_and_scene_state() {
    using namespace interaction;

    const Transform placed{vec3(1.0F, 0.82F, 4.0F), quat()};
    const PlacedSupportContext destination{
        Transform{vec3(0.0F, 0.70F, 4.0F), quat()},
        vec3(2.0F, 0.04F, 0.60F),
    };

    {
        AttachmentFixture fixture = make_fixture();
        AttachmentController attachment(fixture.registry);
        (void)drive_to_held(fixture, attachment);
        InteractionTarget* stored = fixture.registry.find(
            fixture.request.target);
        assert(stored != nullptr);
        ++stored->owner_request;
        const AttachmentSnapshot snapshot = snapshot_attachment(
            fixture, attachment);
        assert(!attachment.commit_place(placed, destination).has_value());
        assert_attachment_unchanged(fixture, attachment, snapshot);

        stored = fixture.registry.find(fixture.request.target);
        assert(stored != nullptr);
        stored->owner_request = fixture.request.request_id;
        const auto recovered = attachment.commit_place(placed, destination);
        assert(recovered.has_value());
        assert(attachment.state() == ObjectState::Free);
        assert(attachment.result() == ResultCode::Succeeded);
        assert(attachment.reason() == Reason::None);
        assert(attachment.held_seconds() == 0.0F);
        assert(exact(attachment.object_world(), placed));
    }

    {
        AttachmentFixture fixture = make_fixture();
        AttachmentController attachment(fixture.registry);
        const Transform held = drive_to_held(fixture, attachment);
        const TargetHandle replacement = fixture.registry.replace_pose(
            fixture.request.target.id, held);
        assert(fixture.registry.reserve(
            replacement, fixture.request.request_id));
        assert(fixture.registry.attach(
            replacement, fixture.request.request_id));
        assert(fixture.registry.hold(
            replacement, fixture.request.request_id));
        const AttachmentSnapshot snapshot = snapshot_attachment(
            fixture, attachment);
        assert(snapshot.registry_target.handle == replacement);
        assert(!attachment.commit_place(placed, destination).has_value());
        assert_attachment_unchanged(fixture, attachment, snapshot);
    }

    {
        AttachmentFixture fixture = make_fixture();
        AttachmentController attachment(fixture.registry);
        (void)drive_to_held(fixture, attachment);
        assert(fixture.registry.release(
            fixture.request.target, fixture.request.request_id));
        const AttachmentSnapshot snapshot = snapshot_attachment(
            fixture, attachment);
        assert(snapshot.registry_target.state == ObjectState::Free);
        assert(!attachment.commit_place(placed, destination).has_value());
        assert_attachment_unchanged(fixture, attachment, snapshot);
    }
}

void test_commit_place_rejects_unstarted_and_attached_controllers() {
    using namespace interaction;

    const Transform placed{vec3(1.0F, 0.82F, 4.0F), quat()};
    const PlacedSupportContext destination{
        Transform{vec3(0.0F, 0.70F, 4.0F), quat()},
        vec3(2.0F, 0.04F, 0.60F),
    };

    {
        AttachmentFixture fixture = make_fixture();
        AttachmentController attachment(fixture.registry);
        const AttachmentSnapshot snapshot = snapshot_attachment(
            fixture, attachment);
        assert(!attachment.commit_place(placed, destination).has_value());
        assert_attachment_unchanged(fixture, attachment, snapshot);
    }

    {
        AttachmentFixture fixture = make_fixture();
        AttachmentController attachment(fixture.registry);
        assert(attachment.begin(
            fixture.target,
            fixture.request,
            fixture.affordance,
            fixture.target.object_world.position.y));
        assert(attachment.try_contact(valid_measurement(fixture)));
        assert(attachment.state() == ObjectState::Attached);
        const AttachmentSnapshot snapshot = snapshot_attachment(
            fixture, attachment);
        assert(!attachment.commit_place(placed, destination).has_value());
        assert_attachment_unchanged(fixture, attachment, snapshot);
    }
}

void test_commit_place_invalid_inputs_and_overflow_are_transactional() {
    using namespace interaction;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Transform valid_placed{vec3(1.0F, 0.82F, 4.0F), quat()};
    const PlacedSupportContext valid_destination{
        Transform{vec3(0.0F, 0.70F, 4.0F), quat()},
        vec3(2.0F, 0.04F, 0.60F),
    };

    {
        AttachmentFixture fixture = make_fixture();
        AttachmentController attachment(fixture.registry);
        (void)drive_to_held(fixture, attachment);
        const AttachmentSnapshot snapshot = snapshot_attachment(
            fixture, attachment);
        Transform invalid = valid_placed;
        invalid.position.x = nan;
        assert(throws_invalid_argument([&] {
            (void)attachment.commit_place(invalid, valid_destination);
        }));
        assert_attachment_unchanged(fixture, attachment, snapshot);
    }

    {
        AttachmentFixture fixture = make_fixture();
        AttachmentController attachment(fixture.registry);
        (void)drive_to_held(fixture, attachment);
        const AttachmentSnapshot snapshot = snapshot_attachment(
            fixture, attachment);
        PlacedSupportContext invalid = valid_destination;
        invalid.table_world.rotation = quat(0.0F, 0.0F, 0.0F, 0.0F);
        assert(throws_invalid_argument([&] {
            (void)attachment.commit_place(valid_placed, invalid);
        }));
        assert_attachment_unchanged(fixture, attachment, snapshot);
    }

    {
        AttachmentFixture fixture = make_fixture();
        AttachmentController attachment(fixture.registry);
        (void)drive_to_held(fixture, attachment);
        const AttachmentSnapshot snapshot = snapshot_attachment(
            fixture, attachment);
        PlacedSupportContext invalid = valid_destination;
        invalid.table_size.z = 0.0F;
        assert(throws_invalid_argument([&] {
            (void)attachment.commit_place(valid_placed, invalid);
        }));
        assert_attachment_unchanged(fixture, attachment, snapshot);
    }

    {
        constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();
        AttachmentFixture fixture = make_fixture(maximum);
        AttachmentController attachment(fixture.registry);
        (void)drive_to_held(fixture, attachment);
        const AttachmentSnapshot snapshot = snapshot_attachment(
            fixture, attachment);
        assert(!attachment.commit_place(
            valid_placed, valid_destination).has_value());
        assert_attachment_unchanged(fixture, attachment, snapshot);
    }
}

void test_repeated_contact_is_rejected_and_reset_can_rebegin() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    assert(attachment.try_contact(valid_measurement(fixture)));
    const Transform attached_pose = attachment.object_world();
    ContactMeasurement duplicate = valid_measurement(fixture);
    duplicate.hand_world.position =
        duplicate.hand_world.position + vec3(3.0F, 4.0F, 5.0F);

    assert(!attachment.try_contact(duplicate));
    assert(attachment.state() == ObjectState::Attached);
    assert(attachment.result() == ResultCode::Accepted);
    assert(attachment.reason() == Reason::None);
    assert(near(attachment.object_world(), attached_pose));
    assert(fixture.registry.find(fixture.request.target)->state ==
           ObjectState::Attached);

    Transform held_pose = attached_pose;
    held_pose.position.y += 0.20F;
    ContactMeasurement held_measurement = measurement_for_object(
        fixture, held_pose);
    attachment.update(held_measurement, 1.0F);
    assert(attachment.state() == ObjectState::Held);
    assert(attachment.result() == ResultCode::Succeeded);
    const float held_seconds = attachment.held_seconds();
    duplicate = held_measurement;
    duplicate.stable_contact_event = true;
    duplicate.hand_world.position =
        duplicate.hand_world.position + vec3(-2.0F, 3.0F, 4.0F);
    assert(!attachment.try_contact(duplicate));
    assert(attachment.state() == ObjectState::Held);
    assert(attachment.result() == ResultCode::Succeeded);
    assert(attachment.reason() == Reason::None);
    assert(near(attachment.object_world(), held_pose));
    assert(near(attachment.held_seconds(), held_seconds));
    assert(fixture.registry.find(fixture.request.target)->state ==
           ObjectState::Held);

    const Transform restored{
        vec3(0.125F, 0.75F, -0.25F),
        quat_from_angle_axis(0.25F, vec3(0.0F, 1.0F, 0.0F)),
    };
    const TargetHandle reset = attachment.reset(restored);
    constexpr uint64_t second_request_id = 7002;
    assert(fixture.registry.reserve(reset, second_request_id));
    const InteractionTarget* second_target = fixture.registry.find(reset);
    const GraspAffordance* second_affordance =
        fixture.registry.find_affordance(reset, fixture.affordance.id);
    assert(second_target != nullptr && second_affordance != nullptr);
    const PickRequest second_request{
        reset, second_affordance->id, second_request_id};

    assert(attachment.begin(
        *second_target,
        second_request,
        *second_affordance,
        restored.position.y));
    assert(attachment.state() == ObjectState::Targeted);
    assert(attachment.result() == ResultCode::Accepted);
    assert(attachment.reason() == Reason::None);
    assert(exact(attachment.object_world(), restored));
    ContactMeasurement second_contact{};
    second_contact.target = reset;
    second_contact.hand = second_affordance->hand;
    second_contact.hand_world = compose(
        restored, second_affordance->hand_in_object);
    second_contact.stable_contact_event = true;
    second_contact.hand_contact = true;
    second_contact.joints_valid = true;
    second_contact.clearance_valid = true;
    assert(attachment.try_contact(second_contact));
    assert(attachment.state() == ObjectState::Attached);
}

void assert_begin_rejected(
    AttachmentFixture& fixture,
    interaction::InteractionTarget target,
    interaction::PickRequest request,
    interaction::GraspAffordance affordance,
    interaction::Reason expected_reason) {
    using namespace interaction;

    AttachmentController attachment(fixture.registry);
    assert(!attachment.begin(
        target, request, affordance, target.object_world.position.y));
    assert(attachment.state() == ObjectState::Free);
    assert(attachment.result() == ResultCode::Rejected);
    assert(attachment.reason() == expected_reason);
    assert(near(attachment.held_seconds(), 0.0F));
}

void test_begin_rejects_incoherent_or_unreserved_inputs() {
    using namespace interaction;

    {
        AttachmentFixture fixture = make_fixture();
        InteractionTarget wrong_profile = fixture.target;
        ++wrong_profile.object_profile_id;
        assert_begin_rejected(
            fixture,
            wrong_profile,
            fixture.request,
            fixture.affordance,
            Reason::TargetChanged);
    }
    {
        AttachmentFixture fixture = make_fixture();
        InteractionTarget wrong_bounds = fixture.target;
        wrong_bounds.object_bounds.center_object.x += 0.001F;
        assert_begin_rejected(
            fixture,
            wrong_bounds,
            fixture.request,
            fixture.affordance,
            Reason::TargetChanged);
    }
    {
        AttachmentFixture fixture = make_fixture();
        InteractionTarget stale = fixture.target;
        ++stale.handle.generation;
        assert_begin_rejected(
            fixture,
            stale,
            fixture.request,
            fixture.affordance,
            Reason::TargetChanged);
    }
    {
        AttachmentFixture fixture = make_fixture();
        PickRequest stale = fixture.request;
        ++stale.target.generation;
        assert_begin_rejected(
            fixture,
            fixture.target,
            stale,
            fixture.affordance,
            Reason::TargetChanged);
    }
    {
        AttachmentFixture fixture = make_fixture();
        PickRequest wrong_owner = fixture.request;
        ++wrong_owner.request_id;
        assert_begin_rejected(
            fixture,
            fixture.target,
            wrong_owner,
            fixture.affordance,
            Reason::TargetChanged);
    }
    {
        AttachmentFixture fixture = make_fixture();
        PickRequest missing_affordance = fixture.request;
        ++missing_affordance.affordance_id;
        assert_begin_rejected(
            fixture,
            fixture.target,
            missing_affordance,
            fixture.affordance,
            Reason::TargetUnavailable);
    }
    {
        AttachmentFixture fixture = make_fixture();
        GraspAffordance unauthored = fixture.affordance;
        ++unauthored.id;
        assert_begin_rejected(
            fixture,
            fixture.target,
            fixture.request,
            unauthored,
            Reason::TargetUnavailable);
    }
    {
        AttachmentFixture fixture = make_fixture();
        assert(fixture.registry.release(
            fixture.request.target, fixture.request.request_id));
        assert_begin_rejected(
            fixture,
            fixture.target,
            fixture.request,
            fixture.affordance,
            Reason::TargetChanged);
    }
}

void test_begin_rejects_authored_slot_metadata_mismatch() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    InteractionTarget* stored = fixture.registry.find(fixture.request.target);
    assert(stored != nullptr);
    stored->affordances.front().interaction_slots = {
        {3U, -0.41F, -0.22F, 1.10F},
        {9U, 0.18F, -0.39F, 0.20F},
    };
    fixture.target = *stored;
    fixture.affordance = stored->affordances.front();

    GraspAffordance copied = fixture.affordance;
    copied.interaction_slots[0].root_x_object_m += 0.001F;
    assert_begin_rejected(
        fixture,
        fixture.target,
        fixture.request,
        copied,
        Reason::TargetUnavailable);
}

void test_invalid_configuration_and_nonfinite_prelift_are_rejected() {
    using namespace interaction;

    auto rejected_config = [](AttachmentConfig config) {
        TargetRegistry registry;
        return throws_invalid_argument([&] {
            AttachmentController attachment(registry, config);
            (void)attachment;
        });
    };

    AttachmentConfig config{};
    config.maximum_position_error_m = -0.001F;
    assert(rejected_config(config));
    config = AttachmentConfig{};
    config.maximum_orientation_error_radians =
        std::numeric_limits<float>::infinity();
    assert(rejected_config(config));
    config = AttachmentConfig{};
    config.required_lift_m = -0.001F;
    assert(rejected_config(config));
    config = AttachmentConfig{};
    config.required_hold_seconds =
        std::numeric_limits<float>::quiet_NaN();
    assert(rejected_config(config));

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    assert(throws_invalid_argument([&] {
        (void)attachment.begin(
            fixture.target,
            fixture.request,
            fixture.affordance,
            std::numeric_limits<float>::quiet_NaN());
    }));
    assert(attachment.state() == ObjectState::Free);
    assert(attachment.result() == ResultCode::None);
    assert(attachment.reason() == Reason::None);
    assert(fixture.registry.validate(
        fixture.request.target, fixture.request.request_id));
}

void test_invalid_contact_scalars_and_transforms_do_not_attach() {
    using namespace interaction;

    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.position_error_m = -0.001F;
        },
        Reason::ContactPosition);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.position_error_m =
                std::numeric_limits<float>::quiet_NaN();
        },
        Reason::ContactPosition);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.orientation_error_radians = -0.001F;
        },
        Reason::ContactOrientation);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.orientation_error_radians =
                std::numeric_limits<float>::infinity();
        },
        Reason::ContactOrientation);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.hand_world.position.x =
                std::numeric_limits<float>::quiet_NaN();
        },
        Reason::ContactPosition);
    expect_contact_gate_failure(
        [](AttachmentFixture&, ContactMeasurement& measurement) {
            measurement.hand_world.rotation = quat(0.0F, 0.0F, 0.0F, 0.0F);
        },
        Reason::ContactOrientation);
}

void test_overflowing_derived_pose_does_not_attach_or_mutate() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    const Transform original = attachment.object_world();
    ContactMeasurement overflow = valid_measurement(fixture);
    const float maximum = std::numeric_limits<float>::max();
    overflow.hand_world.rotation = quat(maximum, maximum, maximum, maximum);

    assert(!attachment.try_contact(overflow));
    assert(attachment.state() == ObjectState::Targeted);
    assert(attachment.result() == ResultCode::Failed);
    assert(attachment.reason() == Reason::ContactOrientation);
    assert(near(attachment.object_world(), original));
    const InteractionTarget* target =
        fixture.registry.find(fixture.request.target);
    assert(target != nullptr);
    assert(target->state == ObjectState::Targeted);
    assert(target->owner_request == fixture.request.request_id);
}

void test_update_rejects_invalid_dt_without_mutating_state() {
    using namespace interaction;

    AttachmentFixture fixture = make_fixture();
    AttachmentController attachment(fixture.registry);
    assert(attachment.begin(
        fixture.target,
        fixture.request,
        fixture.affordance,
        fixture.target.object_world.position.y));
    assert(attachment.try_contact(valid_measurement(fixture)));
    Transform lifted = fixture.target.object_world;
    lifted.position.y += 0.20F;
    const ContactMeasurement measurement = measurement_for_object(
        fixture, lifted);
    const Transform before = attachment.object_world();

    for (const float invalid_dt : {
             -0.001F,
             std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity(),
         }) {
        assert(throws_invalid_argument([&] {
            attachment.update(measurement, invalid_dt);
        }));
        assert(attachment.state() == ObjectState::Attached);
        assert(attachment.result() == ResultCode::Accepted);
        assert(attachment.reason() == Reason::None);
        assert(near(attachment.object_world(), before));
        assert(near(attachment.held_seconds(), 0.0F));
    }

    attachment.update(measurement, 0.0F);
    assert(attachment.state() == ObjectState::Attached);
    assert(near(attachment.object_world(), lifted));
    assert(near(attachment.held_seconds(), 0.0F));
}

}  // namespace

int main() {
    test_frozen_public_contract_and_defaults();
    test_every_contact_gate_rejects_without_teleporting();
    test_valid_contact_attaches_at_inclusive_boundaries();
    test_failed_contact_can_be_remeasured_without_teleporting();
    test_object_follows_grasp_and_hold_requires_continuous_lift();
    test_post_attach_updates_require_contact_but_not_a_second_event();
    test_held_contact_loss_is_terminal_and_freezes_last_valid_state();
    test_generation_change_fails_without_attaching_replacement();
    test_reset_restores_exact_pose_and_increments_generation();
    test_commit_place_atomically_releases_to_destination_support();
    test_commit_place_registry_rejections_preserve_local_and_scene_state();
    test_commit_place_rejects_unstarted_and_attached_controllers();
    test_commit_place_invalid_inputs_and_overflow_are_transactional();
    test_repeated_contact_is_rejected_and_reset_can_rebegin();
    test_begin_rejects_incoherent_or_unreserved_inputs();
    test_begin_rejects_authored_slot_metadata_mismatch();
    test_invalid_configuration_and_nonfinite_prelift_are_rejected();
    test_invalid_contact_scalars_and_transforms_do_not_attach();
    test_overflowing_derived_pose_does_not_attach_or_mutate();
    test_update_rejects_invalid_dt_without_mutating_state();
}
