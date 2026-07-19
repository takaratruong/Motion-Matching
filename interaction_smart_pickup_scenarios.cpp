#include "interaction_smart_pickup_scenarios.h"

#include "g1_skeleton.h"
#include "interaction_smart_pickup_controller.h"
#include "interaction_smart_pickup_scene.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interaction {
namespace {

constexpr float kFixedDt = 0.04F;
constexpr uint64_t kRequestId = 1U;
constexpr uint32_t kMaximumAssistTicks = 240U;
constexpr uint32_t kMaximumRuntimeTicks = 600U;
constexpr float kCarryDisplacementM = 0.02F;
constexpr float kGraspPositionToleranceM = 0.00002F;
constexpr float kGraspOrientationToleranceRadians = 0.00002F;

float planar_yaw(quat rotation) {
    const vec3 forward = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

Transform planar_root(float x, float z, float yaw) {
    return {
        vec3(x, 0.0F, z),
        quat_from_angle_axis(yaw, vec3(0.0F, 1.0F, 0.0F)),
    };
}

InteractionTarget materialize_target(
    const SmartPickupDemoScenario& scenario) {
    InteractionTarget target = make_smart_pickup_demo_target();
    target.table_world = compose(
        scenario.scene_from_authored, target.table_world);
    target.object_world = compose(
        scenario.scene_from_authored, target.object_world);
    return target;
}

std::array<SmartPickupDemoScenario, 7> make_scenarios() {
    std::array<SmartPickupDemoScenario, 7> scenarios{};
    scenarios[0] = {
        "clear_front",
        planar_root(0.0F, 1.60F, 0.0F),
        Transform{},
        {},
        SmartPickupScenarioClass::Clear,
    };
    scenarios[1] = {
        "clear_left",
        planar_root(-0.65F, 2.05F, 0.90F),
        Transform{},
        {},
        SmartPickupScenarioClass::Clear,
    };
    scenarios[2] = {
        "clear_right",
        planar_root(0.65F, 2.05F, -0.90F),
        Transform{},
        {},
        SmartPickupScenarioClass::Clear,
    };
    scenarios[3] = {
        "translated",
        planar_root(0.55F, 1.35F, 0.0F),
        Transform{vec3(0.55F, 0.0F, -0.25F), quat()},
        {},
        SmartPickupScenarioClass::Clear,
    };
    scenarios[4] = {
        "yawed",
        planar_root(1.15F, 1.55F, 0.55F),
        Transform{
            vec3(0.30F, 0.0F, -0.20F),
            quat_from_angle_axis(
                0.55F, vec3(0.0F, 1.0F, 0.0F)),
        },
        {},
        SmartPickupScenarioClass::Clear,
    };
    scenarios[5] = {
        "alternate_slot",
        planar_root(0.0F, 1.60F, 0.0F),
        Transform{},
        {},
        SmartPickupScenarioClass::AlternateSlot,
    };
    scenarios[6] = {
        "all_blocked",
        planar_root(0.0F, 1.60F, 0.0F),
        Transform{},
        {{vec3(0.0F, 0.5F, 2.40F), vec3(3.0F, 1.0F, 3.0F)}},
        SmartPickupScenarioClass::AllBlocked,
    };

    const InteractionTarget target = materialize_target(scenarios[5]);
    const PickSlotSelection nominal = select_pick_slot(
        scenarios[5].initial_root_world,
        target,
        target.affordances.front(),
        {});
    if (!nominal.selected_index.has_value()) {
        throw std::logic_error(
            "alternate Smart Pickup scenario has no nominal route");
    }
    scenarios[5].obstacles.push_back({
        vec3(0.755F, 0.5F, 2.265F),
        vec3(0.002F, 1.0F, 0.002F),
    });
    return scenarios;
}

Transform pose_hand_world(const Pose& pose, Hand hand) {
    const WorldPose world = world_pose(pose);
    const size_t bone = hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
    return {world.positions[bone], world.rotations[bone]};
}

bool transform_near(
    Transform actual,
    Transform expected,
    float position_tolerance_m,
    float orientation_tolerance_radians) {
    const float planar_position_error = std::hypot(
        actual.position.x - expected.position.x,
        actual.position.z - expected.position.z);
    return planar_position_error <= position_tolerance_m &&
        quat_angle_between(actual.rotation, expected.rotation) <=
            orientation_tolerance_radians;
}

LocomotionSnapshot displaced_flat_snapshot(
    const LocomotionSnapshot& published,
    const Pose& pose,
    vec3 displacement) {
    LocomotionSnapshot result = published;
    result.pose = pose;
    result.pose.positions[g1_skeleton::Simulation] =
        result.pose.positions[g1_skeleton::Simulation] + displacement;
    return result;
}

SmartPickupPostStepInput post_step_input(
    const RuntimeOutput& output,
    const LocomotionSnapshot& snapshot,
    const InteractionTarget* target,
    const SmartPickupDemoScenario& scenario) {
    SmartPickupPostStepInput input{};
    input.runtime_state = output.diagnostics.state;
    input.live_flat_snapshot = snapshot;
    input.current_target = target;
    for (const PickNavigationObstacle& obstacle : scenario.obstacles) {
        input.obstacle_centers.push_back(obstacle.center_world);
        input.obstacle_sizes.push_back(obstacle.size_world);
    }
    input.next_request_id = kRequestId;
    return input;
}

}  // namespace

const std::array<SmartPickupDemoScenario, 7>&
smart_pickup_demo_scenarios() {
    static const std::array<SmartPickupDemoScenario, 7> scenarios =
        make_scenarios();
    return scenarios;
}

SmartPickupTargetCompatibility classify_smart_pickup_target(
    const InteractionTarget& target,
    uint32_t affordance_id) {
    const auto found = std::find_if(
        target.affordances.begin(),
        target.affordances.end(),
        [affordance_id](const GraspAffordance& affordance) {
            return affordance.id == affordance_id;
        });
    if (found == target.affordances.end()) {
        return SmartPickupTargetCompatibility::MissingAffordance;
    }
    if (found->hand != Hand::Right) {
        return SmartPickupTargetCompatibility::UnsupportedHand;
    }
    if (found->interaction_slots.empty()) {
        return SmartPickupTargetCompatibility::NoAuthoredSlot;
    }
    return SmartPickupTargetCompatibility::Compatible;
}

const char* smart_pickup_target_compatibility_name(
    SmartPickupTargetCompatibility value) {
    switch (value) {
    case SmartPickupTargetCompatibility::Compatible: return "Compatible";
    case SmartPickupTargetCompatibility::MissingAffordance:
        return "MissingAffordance";
    case SmartPickupTargetCompatibility::UnsupportedHand:
        return "UnsupportedHand";
    case SmartPickupTargetCompatibility::NoAuthoredSlot:
        return "NoAuthoredSlot";
    }
    throw std::invalid_argument(
        "unknown Smart Pickup target compatibility");
}

SmartPickupLifecycleWitness run_smart_pickup_lifecycle(
    const Database& database,
    const Features& features,
    const LocomotionSnapshot& published_flat_snapshot,
    const SmartPickupDemoScenario& scenario,
    const CertifiedPickupSourceRegistry* pickup_source_registry) {
    SmartPickupLifecycleWitness witness{};
    TargetRegistry registry{};
    const TargetHandle target_handle = registry.upsert(
        materialize_target(scenario));
    witness.target_before = target_handle;
    witness.target_after = target_handle;
    witness.terminal_state = RuntimeState::Locomotion;
    const InteractionTarget* initial_target = registry.find(target_handle);
    if (initial_target == nullptr || initial_target->affordances.empty()) {
        return witness;
    }
    const GraspAffordance affordance = initial_target->affordances.front();
    const PickSlotSelection selection = select_pick_slot(
        scenario.initial_root_world,
        *initial_target,
        affordance,
        scenario.obstacles);
    if (!selection.selected_index.has_value()) return witness;
    const Transform selected_root =
        selection.ordered[*selection.selected_index].root_world;
    const runtime_detail::PickSnapshotMap mapped =
        runtime_detail::map_pick_entry_snapshot(
            published_flat_snapshot,
            PickEntryRoot{
                selected_root.position.x,
                selected_root.position.z,
                planar_yaw(selected_root.rotation),
            });
    if (!mapped.accepted) return witness;

    InteractionRuntime runtime(
        database,
        features,
        registry,
        RuntimeConfig{},
        pickup_source_registry);
    SmartPickupController controller{};
    RuntimeOutput output{};
    output.pose = mapped.snapshot.pose;
    output.object_world = initial_target->object_world;
    output.diagnostics = runtime.diagnostics();

    const auto update_runtime = [&](const RuntimeInput& input) {
        const bool was_attached = output.diagnostics.attached;
        output = runtime.update(input);
        if (!was_attached && output.diagnostics.attached) {
            ++witness.attachment_edges;
        }
        if (was_attached && !output.diagnostics.attached) {
            ++witness.release_edges;
        }
    };

    SmartPickupPreStepInput activation{};
    activation.runtime_state = RuntimeState::Locomotion;
    activation.interact_pressed = true;
    activation.selected_target = registry.find(target_handle);
    activation.selected_affordance_id = affordance.id;
    (void)controller.pre_step(activation);
    SmartPickupPostStepResult assist = controller.post_step(
        post_step_input(
            output,
            mapped.snapshot,
            registry.find(target_handle),
            scenario),
        {});

    bool request_submitted = false;
    for (uint32_t tick = 0U;
         tick < kMaximumAssistTicks && !request_submitted;
         ++tick) {
        SmartPickupPreStepInput pre{};
        pre.runtime_state = output.diagnostics.state;
        pre.selected_target = registry.find(target_handle);
        pre.selected_affordance_id = affordance.id;
        (void)controller.pre_step(pre);
        assist = controller.post_step(
            post_step_input(
                output,
                mapped.snapshot,
                registry.find(target_handle),
                scenario),
            [&](const LocomotionSnapshot& snapshot,
                PickEntryRoot root,
                TargetHandle target,
                uint32_t affordance_id)
                -> std::optional<PickEntryPreview> {
                return runtime.preview_pick(
                    snapshot, root, target, affordance_id);
            });

        RuntimeInput input{};
        input.dt = kFixedDt;
        input.locomotion = mapped.snapshot;
        if (assist.pick_request.has_value()) {
            ++witness.request_count;
            witness.request_id = assist.pick_request->request_id;
            input.interact_pressed = true;
            input.pick_request = assist.pick_request;
            request_submitted = true;
        }
        update_runtime(input);
    }

    for (uint32_t tick = 0U;
         request_submitted && tick < kMaximumRuntimeTicks;
         ++tick) {
        if (output.diagnostics.state == RuntimeState::Carry &&
            output.diagnostics.attached) {
            const Pose carry_pose_before = output.pose;
            const Transform carry_object_before = output.object_world;
            RuntimeInput carry_tick{};
            carry_tick.dt = kFixedDt;
            carry_tick.locomotion = displaced_flat_snapshot(
                mapped.snapshot,
                output.pose,
                vec3(kCarryDisplacementM, 0.0F, 0.0F));
            update_runtime(carry_tick);
            const Transform expected_hand_after = compose(
                output.object_world, affordance.hand_in_object);
            const Transform actual_hand_after = pose_hand_world(
                output.pose, affordance.hand);
            witness.post_carry_tick_observed = true;
            witness.grasp_preserved = transform_near(
                actual_hand_after,
                expected_hand_after,
                kGraspPositionToleranceM,
                kGraspOrientationToleranceRadians);
            witness.carry_root_displacement_m = std::hypot(
                output.pose.positions[g1_skeleton::Simulation].x -
                    carry_pose_before.positions[g1_skeleton::Simulation].x,
                output.pose.positions[g1_skeleton::Simulation].z -
                    carry_pose_before.positions[g1_skeleton::Simulation].z);
            witness.carry_object_displacement_m = std::hypot(
                output.object_world.position.x -
                    carry_object_before.position.x,
                output.object_world.position.z -
                    carry_object_before.position.z);
            break;
        }
        RuntimeInput input{};
        input.dt = kFixedDt;
        input.locomotion = mapped.snapshot;
        update_runtime(input);
        if (output.diagnostics.state == RuntimeState::Locomotion &&
            (output.diagnostics.result == ResultCode::Rejected ||
             output.diagnostics.result == ResultCode::Failed)) {
            break;
        }
    }

    const InteractionTarget* authoritative = registry.find(target_handle);
    if (authoritative == nullptr) {
        authoritative = registry.find_by_id(target_handle.id);
    }
    if (authoritative != nullptr) {
        witness.target_after = authoritative->handle;
        witness.owner_request = authoritative->owner_request;
        witness.terminal_object_state = authoritative->state;
    } else {
        witness.terminal_object_state = output.diagnostics.object_state;
    }
    witness.terminal_state = output.diagnostics.state;
    witness.attached = output.diagnostics.attached;
    witness.actual_source = output.diagnostics.pickup_source;
    return witness;
}

}  // namespace interaction
