#include "interaction_controller_adapter.h"

#include <algorithm>
#include <cstddef>

namespace interaction {
namespace {

constexpr int kControllerRate = 60;
constexpr int kInteractionRuntimeRate = 25;
constexpr float kOwnershipBlendSeconds = 0.25F;

Transform frame_transform(
    const std::vector<float>& positions,
    const std::vector<float>& rotations,
    size_t index) {
    return {
        vec3(
            positions.at(index * 3U),
            positions.at(index * 3U + 1U),
            positions.at(index * 3U + 2U)),
        quat(
            rotations.at(index * 4U),
            rotations.at(index * 4U + 1U),
            rotations.at(index * 4U + 2U),
            rotations.at(index * 4U + 3U))};
}

vec3 clip_vector(const std::vector<float>& values, size_t clip) {
    return vec3(
        values.at(clip * 3U),
        values.at(clip * 3U + 1U),
        values.at(clip * 3U + 2U));
}

Transform clip_transform(
    const std::vector<float>& positions,
    const std::vector<float>& rotations,
    size_t clip) {
    return frame_transform(positions, rotations, clip);
}

Pose blend_pose(const Pose& source, const Pose& target, float alpha) {
    Pose blended;
    for (size_t bone = 0; bone < blended.positions.size(); ++bone) {
        blended.positions[bone] =
            lerp(source.positions[bone], target.positions[bone], alpha);
        blended.velocities[bone] =
            lerp(source.velocities[bone], target.velocities[bone], alpha);
        blended.rotations[bone] = quat_nlerp_shortest(
            source.rotations[bone], target.rotations[bone], alpha);
        blended.angular_velocities[bone] = lerp(
            source.angular_velocities[bone],
            target.angular_velocities[bone],
            alpha);
    }
    for (size_t joint = 0; joint < blended.hand_dof.size(); ++joint) {
        blended.hand_dof[joint] =
            lerpf(source.hand_dof[joint], target.hand_dof[joint], alpha);
        blended.hand_dof_velocities[joint] = lerpf(
            source.hand_dof_velocities[joint],
            target.hand_dof_velocities[joint],
            alpha);
    }
    blended.foot_contacts =
        alpha < 0.5F ? source.foot_contacts : target.foot_contacts;
    return blended;
}

}  // namespace

const RuntimeOutput& ControllerInteractionScheduler::tick(
    ControllerInteractionEdges edges,
    const LocomotionProvider& locomotion_provider,
    const PickRequestResolver& request_resolver,
    const RuntimeUpdate& runtime_update) {
    pending_interact_ = pending_interact_ || edges.interact_pressed;
    pending_cancel_ = pending_cancel_ || edges.cancel_pressed;
    pending_reset_ = pending_reset_ || edges.reset_pressed;

    phase_ += kInteractionRuntimeRate;
    if (phase_ < kControllerRate) {
        return cached_output_;
    }
    phase_ -= kControllerRate;

    RuntimeInput input;
    input.dt = kInteractionRuntimeStepSeconds;
    input.locomotion = locomotion_provider();
    input.interact_pressed = pending_interact_;
    input.cancel_pressed = pending_cancel_;
    input.reset_pressed = pending_reset_;
    if (pending_interact_) {
        input.pick_request = request_resolver(input.locomotion);
    }

    RuntimeOutput next_output = runtime_update(input);
    cached_output_ = std::move(next_output);
    pending_interact_ = false;
    pending_cancel_ = false;
    pending_reset_ = false;
    return cached_output_;
}

int ControllerInteractionScheduler::phase() const {
    return phase_;
}

const RuntimeOutput& ControllerInteractionScheduler::cached_output() const {
    return cached_output_;
}

Pose ControllerInteractionAdapter::apply(
    const Pose& locomotion_pose,
    const RuntimeOutput& runtime_output,
    float dt) {
    if (!runtime_output.owns_pose) {
        owned_last_update_ = false;
        blend_seconds_ = 0.0F;
        return locomotion_pose;
    }

    if (!owned_last_update_) {
        blend_source_ = locomotion_pose;
        blend_seconds_ = 0.0F;
        owned_last_update_ = true;
    }

    blend_seconds_ += std::max(dt, 0.0F);
    if (blend_seconds_ >= kOwnershipBlendSeconds) {
        return runtime_output.pose;
    }
    const float alpha = std::clamp(
        blend_seconds_ / kOwnershipBlendSeconds, 0.0F, 1.0F);
    return blend_pose(blend_source_, runtime_output.pose, alpha);
}

void ControllerInteractionAdapter::reset() {
    owned_last_update_ = false;
    blend_seconds_ = 0.0F;
    blend_source_ = {};
}

ControllerInteractionFrameState ControllerInteractionFrameHandoff::apply(
    const Pose& locomotion_pose,
    const RuntimeOutput& runtime_output,
    float dt) {
    ControllerInteractionFrameState state;
    state.pose = adapter_.apply(locomotion_pose, runtime_output, dt);
    state.owns_pose = runtime_output.owns_pose;
    state.synchronize_simulation_root = runtime_output.owns_pose;
    state.simulation_root_position = state.pose.positions[0];
    state.simulation_root_rotation = state.pose.rotations[0];
    return state;
}

void ControllerInteractionFrameHandoff::reset() {
    adapter_.reset();
}

ControllerInteractionSceneState ControllerInteractionSceneHandoff::apply(
    const InteractionTarget* registry_target,
    const RuntimeOutput& runtime_output,
    const Transform& authored_fallback) {
    if (registry_target == nullptr) {
        has_runtime_pose_ = false;
        runtime_target_ = {};
        return {authored_fallback, false};
    }

    if (registry_target->state == ObjectState::Free ||
        registry_target->state == ObjectState::Targeted) {
        has_runtime_pose_ = false;
        runtime_target_ = {};
        return {registry_target->object_world, false};
    }

    if (runtime_output.diagnostics.target == registry_target->handle) {
        has_runtime_pose_ = true;
        runtime_target_ = registry_target->handle;
        runtime_object_world_ = runtime_output.object_world;
    }
    if (has_runtime_pose_ && runtime_target_ == registry_target->handle) {
        return {runtime_object_world_, true};
    }
    return {registry_target->object_world, false};
}

const char* controller_carry_mode_label(const RuntimeOutput& output) {
    if (output.diagnostics.state != RuntimeState::Carry) {
        return "none";
    }
    return output.diagnostics.recorded_carry ? "recorded" : "layered";
}

InteractionTarget make_controller_demo_target(const Database& database) {
    if (database.clip_count == 0U || database.range_starts.empty() ||
        database.range_stops.empty()) {
        throw FormatError("interaction database has no clip 0");
    }

    const int32_t start = database.range_starts.at(0);
    const int32_t stop = database.range_stops.at(0);
    int32_t contact = -1;
    for (int32_t frame = start; frame < stop; ++frame) {
        if (database.phases.at(static_cast<size_t>(frame)) ==
            static_cast<uint8_t>(Phase::Contact)) {
            contact = frame;
            break;
        }
    }
    if (contact <= start) {
        throw FormatError("clip 0 has no pre-contact object sample");
    }

    const Transform source_table = clip_transform(
        database.table_positions, database.table_rotations, 0U);
    const Transform source_object = frame_transform(
        database.object_positions,
        database.object_rotations,
        static_cast<size_t>(contact - 1));
    const Transform object_in_table = compose(inverse(source_table), source_object);

    InteractionTarget target;
    target.handle = {1U, 1U};
    target.table_world = source_table;
    target.table_world.position.x = 0.0F;
    target.table_world.position.z = 3.0F;
    target.object_world = compose(target.table_world, object_in_table);
    target.table_size = clip_vector(database.table_sizes, 0U);
    target.object_dimensions = clip_vector(database.object_dimensions, 0U);
    target.state = ObjectState::Free;

    GraspAffordance affordance;
    affordance.id = 1U;
    affordance.hand = static_cast<Hand>(database.active_hands.at(0));
    affordance.hand_in_object = clip_transform(
        database.grasp_positions_object,
        database.grasp_rotations_object,
        0U);
    affordance.approach_direction_object =
        clip_vector(database.approach_directions_object, 0U);
    target.affordances.push_back(affordance);
    return target;
}

void validate_controller_interaction_pack(
    const Database& database,
    const Features& features) {
    if (database.frame_count != features.frame_count) {
        throw FormatError(
            "interaction database/features frame count mismatch");
    }
}

}  // namespace interaction
