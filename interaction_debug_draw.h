#pragma once

#include "interaction_controller_adapter.h"
#include "raylib.h"

#include <array>

namespace interaction::debug_draw {

inline Vector3 ray_vector(vec3 value) {
    return Vector3{value.x, value.y, value.z};
}

inline vec3 world_point(const Transform& transform, vec3 local) {
    return transform.position + quat_mul_vec3(transform.rotation, local);
}

inline void draw_axes(
    const Transform& transform,
    float scale = 0.18F) {
    DrawLine3D(
        ray_vector(transform.position),
        ray_vector(world_point(transform, vec3(scale, 0.0F, 0.0F))),
        RED);
    DrawLine3D(
        ray_vector(transform.position),
        ray_vector(world_point(transform, vec3(0.0F, scale, 0.0F))),
        GREEN);
    DrawLine3D(
        ray_vector(transform.position),
        ray_vector(world_point(transform, vec3(0.0F, 0.0F, scale))),
        BLUE);
}

inline void draw_oriented_box(
    const Transform& transform,
    vec3 dimensions,
    Color fill,
    Color wire) {
    const vec3 half = dimensions * 0.5F;
    const std::array<vec3, 8> local = {
        vec3(-half.x, -half.y, -half.z),
        vec3(+half.x, -half.y, -half.z),
        vec3(+half.x, +half.y, -half.z),
        vec3(-half.x, +half.y, -half.z),
        vec3(-half.x, -half.y, +half.z),
        vec3(+half.x, -half.y, +half.z),
        vec3(+half.x, +half.y, +half.z),
        vec3(-half.x, +half.y, +half.z)};
    std::array<Vector3, 8> corners{};
    for (size_t index = 0; index < corners.size(); ++index) {
        corners[index] = ray_vector(world_point(transform, local[index]));
    }

    constexpr std::array<std::array<int, 4>, 6> faces = {{
        {{0, 1, 2, 3}},
        {{4, 7, 6, 5}},
        {{0, 4, 5, 1}},
        {{3, 2, 6, 7}},
        {{0, 3, 7, 4}},
        {{1, 5, 6, 2}},
    }};
    for (const auto& face : faces) {
        DrawTriangle3D(
            corners[face[0]], corners[face[1]], corners[face[2]], fill);
        DrawTriangle3D(
            corners[face[0]], corners[face[2]], corners[face[3]], fill);
    }

    constexpr std::array<std::array<int, 2>, 12> edges = {{
        {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
        {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
        {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
    }};
    for (const auto& edge : edges) {
        DrawLine3D(corners[edge[0]], corners[edge[1]], wire);
    }
}

inline const char* state_name(RuntimeState state) {
    switch (state) {
    case RuntimeState::Disabled: return "Disabled";
    case RuntimeState::Locomotion: return "Locomotion";
    case RuntimeState::Preflight: return "Preflight";
    case RuntimeState::Align: return "Align";
    case RuntimeState::PickupReplay: return "PickupReplay";
    case RuntimeState::Hold: return "Hold";
    case RuntimeState::Carry: return "Carry";
    }
    return "Unknown";
}

inline const char* result_name(ResultCode result) {
    switch (result) {
    case ResultCode::None: return "None";
    case ResultCode::Accepted: return "Accepted";
    case ResultCode::Succeeded: return "Succeeded";
    case ResultCode::Rejected: return "Rejected";
    case ResultCode::Cancelled: return "Cancelled";
    case ResultCode::Failed: return "Failed";
    case ResultCode::Reset: return "Reset";
    }
    return "Unknown";
}

inline const char* reason_name(Reason reason) {
    switch (reason) {
    case Reason::None: return "None";
    case Reason::PackUnavailable: return "PackUnavailable";
    case Reason::TargetUnavailable: return "TargetUnavailable";
    case Reason::TargetChanged: return "TargetChanged";
    case Reason::OutOfRange: return "OutOfRange";
    case Reason::NoCandidate: return "NoCandidate";
    case Reason::PoorMatch: return "PoorMatch";
    case Reason::BlockedPath: return "BlockedPath";
    case Reason::CorrectionLimit: return "CorrectionLimit";
    case Reason::Cancelled: return "Cancelled";
    case Reason::ContactPosition: return "ContactPosition";
    case Reason::ContactOrientation: return "ContactOrientation";
    case Reason::JointLimit: return "JointLimit";
    case Reason::LostContact: return "LostContact";
    case Reason::ClipEnded: return "ClipEnded";
    case Reason::Reset: return "Reset";
    case Reason::SurfaceUnavailable: return "SurfaceUnavailable";
    case Reason::SurfaceChanged: return "SurfaceChanged";
    case Reason::PlacementOutOfBounds: return "PlacementOutOfBounds";
    case Reason::ReleasePosition: return "ReleasePosition";
    case Reason::ReleaseOrientation: return "ReleaseOrientation";
    }
    return "Unknown";
}

inline const char* object_state_name(ObjectState state) {
    switch (state) {
    case ObjectState::Free: return "Free";
    case ObjectState::Targeted: return "Targeted";
    case ObjectState::Attached: return "Attached";
    case ObjectState::Held: return "Held";
    }
    return "Unknown";
}

inline void draw_interaction_scene(
    const InteractionTarget* target,
    const Transform& object_world,
    const RuntimeOutput& output,
    const std::array<vec3, 3>& predicted_roots,
    const Pose& displayed_pose,
    const Pose& locomotion_pose,
    float maximum_approach_m) {
    for (size_t index = 0; index < predicted_roots.size(); ++index) {
        DrawSphereWires(ray_vector(predicted_roots[index]), 0.035F, 4, 8, PURPLE);
        if (index != 0U) {
            DrawLine3D(
                ray_vector(predicted_roots[index - 1U]),
                ray_vector(predicted_roots[index]),
                PURPLE);
        }
    }

    if (target == nullptr) {
        return;
    }

    draw_oriented_box(
        target->table_world,
        target->table_size,
        Fade(LIGHTGRAY, 0.35F),
        GRAY);
    draw_oriented_box(
        object_world,
        target->object_dimensions,
        Fade(output.diagnostics.attached ? GOLD : VIOLET, 0.55F),
        output.diagnostics.attached ? ORANGE : PURPLE);
    draw_axes(target->table_world, 0.24F);
    draw_axes(object_world, 0.16F);
    DrawSphereWires(ray_vector(object_world.position), 0.12F, 8, 12, YELLOW);
    DrawCylinderWires(
        ray_vector(vec3(
            object_world.position.x,
            0.002F,
            object_world.position.z)),
        maximum_approach_m,
        maximum_approach_m,
        0.004F,
        32,
        Fade(SKYBLUE, 0.7F));

    if (!target->affordances.empty()) {
        const GraspAffordance& affordance = target->affordances.front();
        const Transform grasp_world = compose(object_world, affordance.hand_in_object);
        draw_axes(grasp_world, 0.14F);
        const vec3 approach_world = quat_mul_vec3(
            object_world.rotation,
            affordance.approach_direction_object);
        DrawLine3D(
            ray_vector(grasp_world.position),
            ray_vector(grasp_world.position + 0.28F * approach_world),
            MAGENTA);

        const WorldPose pose_world = world_pose(displayed_pose);
        const size_t hand_bone = affordance.hand == Hand::Left
            ? kLeftHandBone
            : kRightHandBone;
        const vec3 hand_world = pose_world.positions[hand_bone];
        DrawSphereWires(ray_vector(hand_world), 0.045F, 6, 10, LIME);
        DrawLine3D(
            ray_vector(hand_world),
            ray_vector(grasp_world.position),
            Fade(LIME, 0.75F));
    }

    const vec3 locomotion_root = locomotion_pose.positions[0];
    const vec3 applied_root_delta =
        displayed_pose.positions[0] - locomotion_pose.positions[0];
    if (output.diagnostics.requested_root_correction_m > 0.0F) {
        DrawCylinderWires(
            ray_vector(locomotion_root),
            output.diagnostics.requested_root_correction_m,
            output.diagnostics.requested_root_correction_m,
            0.004F,
            24,
            RED);
    }
    DrawLine3D(
        ray_vector(locomotion_root + vec3(0.0F, 0.025F, 0.0F)),
        ray_vector(
            locomotion_root + vec3(0.0F, 0.025F, 0.0F) +
            applied_root_delta),
        GREEN);
}

inline void draw_interaction_text(
    const RuntimeOutput& output,
    const char* pack_diagnostic,
    int x,
    int y) {
    DrawText("Interaction: F pick  X cancel  R reset", x, y, 18, DARKPURPLE);
    DrawText(
        TextFormat(
            "state=%s result=%s reason=%s clip=%d frame=%d",
            state_name(output.diagnostics.state),
            result_name(output.diagnostics.result),
            reason_name(output.diagnostics.reason),
            output.diagnostics.clip,
            output.diagnostics.frame),
        x,
        y + 22,
        16,
        DARKGRAY);
    DrawText(
        TextFormat(
            "runtime=25Hz controller=25Hz owns=%d attached=%d speed=%.3f",
            output.owns_pose ? 1 : 0,
            output.diagnostics.attached ? 1 : 0,
            output.diagnostics.playback_speed),
        x,
        y + 42,
        16,
        DARKGRAY);
    DrawText(
        TextFormat(
            "target=%llu:%u affordance=%u hand=%s",
            static_cast<unsigned long long>(output.diagnostics.target.id),
            output.diagnostics.target.generation,
            output.diagnostics.affordance_id,
            output.diagnostics.hand == Hand::Left ? "left" : "right"),
        x,
        y + 62,
        16,
        DARKGRAY);
    DrawText(
        TextFormat(
            "cost=%.3f groups=[%.3f %.3f %.3f %.3f %.3f]",
            output.diagnostics.total_cost,
            output.diagnostics.group_costs[0],
            output.diagnostics.group_costs[1],
            output.diagnostics.group_costs[2],
            output.diagnostics.group_costs[3],
            output.diagnostics.group_costs[4]),
        x,
        y + 82,
        16,
        DARKGRAY);
    DrawText(
        TextFormat(
            "root req/app=%.3f/%.3f yaw req/app=%.3f/%.3f",
            output.diagnostics.requested_root_correction_m,
            output.diagnostics.applied_root_correction_m,
            output.diagnostics.requested_yaw_correction_radians,
            output.diagnostics.applied_yaw_correction_radians),
        x,
        y + 102,
        16,
        DARKGRAY);
    DrawText(
        TextFormat(
            "hand pos/orient=%.3f/%.3f object=%s carry=%s",
            output.diagnostics.hand_position_error_m,
            output.diagnostics.hand_orientation_error_radians,
            object_state_name(output.diagnostics.object_state),
            controller_carry_mode_label(output)),
        x,
        y + 122,
        16,
        DARKGRAY);
    if (pack_diagnostic != nullptr && pack_diagnostic[0] != '\0') {
        DrawText(pack_diagnostic, x, y + 142, 14, MAROON);
    }
}

}  // namespace interaction::debug_draw
